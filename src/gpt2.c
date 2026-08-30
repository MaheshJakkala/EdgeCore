#include "edgecore.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* .ecm header parsing                                                 */
/* ------------------------------------------------------------------ */

static const char *readline(const char *p, const char *end, char *line, size_t cap)
{
    size_t n = 0;
    while (p < end && *p != '\n' && n + 1 < cap) {
        if (*p != '\r') line[n++] = *p;
        p++;
    }
    line[n] = '\0';
    if (p < end && *p == '\n') p++;
    return (n == 0 && p >= end) ? NULL : p;
}

static size_t dtype_size(int dtype)
{
    switch (dtype) {
    case 0: return 4; /* fp32 */
    case 1: return 2; /* fp16 */
    case 2: return 1; /* int8 */
    default: return 0;
    }
}

int ecm_parse(const void *data, size_t size, EcmTensor *out, int max,
              GPT2 *meta)
{
    const char *p = data;
    const char *end = (const char *)data + size;
    char line[1024];

    if (size < 5 || memcmp(p, ECM_MAGIC, 5) != 0) {
        fprintf(stderr, "ecm: bad magic\n");
        return -1;
    }
    p += 5;

    while (1) {
        p = readline(p, end, line, sizeof line);
        if (!p) return -1;
        if (strcmp(line, "END_META") == 0) break;
        char *eq = strchr(line, '=');
        if (!eq) continue;
        *eq = '\0';
        const char *key = line, *val = eq + 1;
        if (!strcmp(key, "model_name")) snprintf(meta->model_name, sizeof meta->model_name, "%s", val);
        else if (!strcmp(key, "n_layer")) meta->n_layer = atoi(val);
        else if (!strcmp(key, "n_head")) meta->n_head = atoi(val);
        else if (!strcmp(key, "n_embd")) meta->n_embd = atoi(val);
        else if (!strcmp(key, "n_ctx")) meta->n_ctx = atoi(val);
        else if (!strcmp(key, "vocab")) meta->vocab = atoi(val);
        else if (!strcmp(key, "n_params")) meta->n_params = (size_t)strtoull(val, NULL, 10);
        else if (!strcmp(key, "eps")) meta->eps = atof(val);
    }

    int ntensors = 0;
    while (1) {
        p = readline(p, end, line, sizeof line);
        if (!p) return -1;
        if (strcmp(line, "END_TENSORS") == 0) break;
        EcmTensor *t = &out[ntensors];
        memset(t, 0, sizeof *t);
        char name[128], dtype_s[16];
        int nf = sscanf(line, "%127s %15s", name, dtype_s);
        if (nf < 2) continue;
        snprintf(t->name, sizeof t->name, "%s", name);
        if (!strcmp(dtype_s, "fp32")) t->dtype = 0;
        else if (!strcmp(dtype_s, "fp16")) t->dtype = 1;
        else if (!strcmp(dtype_s, "int8")) t->dtype = 2;
        else if (!strcmp(dtype_s, "text")) t->dtype = 3;
        else continue;

        const char *rest = line;
        int tok = 0;
        while (*rest) {
            while (*rest == ' ') rest++;
            if (tok >= 2) break;
            while (*rest && *rest != ' ') rest++;
            tok++;
        }
        if (t->dtype == 3) {
            t->ndims = 1;
            t->dims[0] = (size_t)strtoull(rest, NULL, 10);
            t->nelems = t->dims[0];
            t->bytes = t->dims[0];
        } else {
            t->ndims = 0;
            t->nelems = 1;
            const char *q = rest;
            while (*q) {
                while (*q == ' ') q++;
                if (!*q) break;
                char *endp;
                t->dims[t->ndims++] = (size_t)strtoull(q, &endp, 10);
                t->nelems *= t->dims[t->ndims - 1];
                q = endp;
            }
            t->bytes = t->nelems * dtype_size(t->dtype);
        }
        ntensors++;
        if (ntensors >= max) return -1;
    }

    /* assign offsets sequentially from the data region */
    size_t off = (size_t)(p - (const char *)data);
    for (int i = 0; i < ntensors; i++) {
        out[i].offset = off;
        off += out[i].bytes;
    }
    return ntensors;
}

static const EcmTensor *find_tensor(const EcmTensor *t, int n, const char *name)
{
    for (int i = 0; i < n; i++)
        if (!strcmp(t[i].name, name)) return &t[i];
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Load                                                                */
/* ------------------------------------------------------------------ */

int gpt2_load(GPT2 *g, const char *path, int batch_cap)
{
    memset(g, 0, sizeof(*g));
    g->batch_cap = batch_cap < 1 ? 1 : batch_cap;
    g->eps = 1e-5;

    /* mmap the file so only touched pages are resident; weights point into it. */
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open model"); return -1; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return -1; }
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) { perror("mmap model"); return -1; }
    g->file_data = map;
    g->file_size = (size_t)st.st_size;
    g->file_mapped = 1;

    EcmTensor tens[512];
    int n = ecm_parse(g->file_data, g->file_size, tens, 512, g);
    if (n < 0) return -1;

    const char *base = g->file_data;

    const EcmTensor *wte = find_tensor(tens, n, "wte");
    const EcmTensor *wpe = find_tensor(tens, n, "wpe");
    const EcmTensor *lnf_g = find_tensor(tens, n, "ln_f.g");
    const EcmTensor *lnf_b = find_tensor(tens, n, "ln_f.b");
    if (!wte || !wpe || !lnf_g || !lnf_b) {
        fprintf(stderr, "ecm: missing core tensors\n");
        return -1;
    }
    g->wte = (const uint16_t *)(base + wte->offset);
    g->wpe = (const uint16_t *)(base + wpe->offset);
    g->lnf_g = (const float *)(base + lnf_g->offset);
    g->lnf_b = (const float *)(base + lnf_b->offset);

    /* precision is determined by the file: the linear weights are stored either
     * fp16 (native fp16 artifact) or int8 (per-channel quantization artifact).
     * There is no "dequantize int8 to fp16" fallback. */
    const EcmTensor *t0 = find_tensor(tens, n, "blocks.0.attn.w");
    if (!t0) {
        fprintf(stderr, "ecm: missing blocks.0.attn.w\n");
        return -1;
    }
    if (t0->dtype == 1) {
        g->precision = 1;   /* fp16 */
    } else if (t0->dtype == 2) {
        g->precision = 0;   /* int8 */
    } else {
        fprintf(stderr, "ecm: unexpected linear weight dtype\n");
        return -1;
    }

    g->lw = calloc((size_t)g->n_layer, sizeof *g->lw);
    if (!g->lw) return -1;

    for (int l = 0; l < g->n_layer; l++) {
        char prefix[64];
        snprintf(prefix, sizeof prefix, "blocks.%d", l);
        char nm[256];
        struct LayerW *W = &g->lw[l];
        snprintf(nm, sizeof nm, "%s.ln_1.g", prefix); { const EcmTensor *t = find_tensor(tens, n, nm); if (t) W->ln1_g = (const float *)(base + t->offset); }
        snprintf(nm, sizeof nm, "%s.ln_1.b", prefix); { const EcmTensor *t = find_tensor(tens, n, nm); if (t) W->ln1_b = (const float *)(base + t->offset); }
        snprintf(nm, sizeof nm, "%s.ln_2.g", prefix); { const EcmTensor *t = find_tensor(tens, n, nm); if (t) W->ln2_g = (const float *)(base + t->offset); }
        snprintf(nm, sizeof nm, "%s.ln_2.b", prefix); { const EcmTensor *t = find_tensor(tens, n, nm); if (t) W->ln2_b = (const float *)(base + t->offset); }

        const char *suff[] = {"attn", "proj", "fc", "fc2"};
        const void **wp[4];
        const float **sp[4];
        const float **bp[4];
        wp[0] = &W->attn_w; wp[1] = &W->proj_w; wp[2] = &W->fc_w; wp[3] = &W->fc2_w;
        sp[0] = &W->attn_s; sp[1] = &W->proj_s; sp[2] = &W->fc_s; sp[3] = &W->fc2_s;
        bp[0] = &W->attn_b; bp[1] = &W->proj_b; bp[2] = &W->fc_b; bp[3] = &W->fc2_b;

        for (int s = 0; s < 4; s++) {
            snprintf(nm, sizeof nm, "%s.%s.w", prefix, suff[s]);
            const EcmTensor *tw = find_tensor(tens, n, nm);
            snprintf(nm, sizeof nm, "%s.%s.s", prefix, suff[s]);
            const EcmTensor *ts = find_tensor(tens, n, nm);
            snprintf(nm, sizeof nm, "%s.%s.b", prefix, suff[s]);
            const EcmTensor *tb = find_tensor(tens, n, nm);
            if (!tw || !tb) {
                fprintf(stderr, "ecm: missing %s tensors\n", nm);
                return -1;
            }
            *wp[s] = base + tw->offset;
            if (g->precision == 0) {
                if (!ts) {
                    fprintf(stderr, "ecm: missing int8 scale %s\n", nm);
                    return -1;
                }
                *sp[s] = (const float *)(base + ts->offset);
            } else {
                *sp[s] = NULL;
            }
            *bp[s] = (const float *)(base + tb->offset);
        }
    }

    /* KV cache */
    size_t kv = (size_t)g->n_layer * (size_t)g->batch_cap * (size_t)g->n_ctx * (size_t)g->n_embd;
    g->k_cache = calloc(kv, sizeof(uint16_t));
    g->v_cache = calloc(kv, sizeof(uint16_t));
    if (!g->k_cache || !g->v_cache) return -1;

    /* workspace: sized for the largest M = max(n_ctx, batch_cap) so that
     * prefill (M up to n_ctx) and batched decode (M up to batch_cap) fit. */
    size_t E = (size_t)g->n_embd;
    size_t B = (size_t)g->batch_cap;
    size_t Mmax = (size_t)g->n_ctx > B ? (size_t)g->n_ctx : B;
    g->ws_x = malloc(Mmax * E * sizeof(float));
    g->ws_tmp = malloc(Mmax * (4 * E) * sizeof(float));
    /* ws_qkv holds the qkv output [M][3E] AND is reused as the MLP fc
     * output / fc2 input [M][4E], so it must be sized for 4E. */
    g->ws_qkv = malloc(Mmax * (4 * E) * sizeof(float));
    g->ws_attn = malloc(Mmax * E * sizeof(float));
    g->ws_scores = malloc((size_t)g->n_ctx * sizeof(float));
    g->ws_q = malloc(Mmax * (4 * E) * sizeof(int8_t));
    g->ws_qs = malloc(Mmax * sizeof(float));
    if (!g->ws_x || !g->ws_tmp || !g->ws_qkv || !g->ws_attn || !g->ws_scores ||
        !g->ws_q || !g->ws_qs)
        return -1;

    /* tokenizer */
    const EcmTensor *tvocab = find_tensor(tens, n, "tokenizer.vocab");
    const EcmTensor *tmerges = find_tensor(tens, n, "tokenizer.merges");
    if (!tvocab || !tmerges) {
        fprintf(stderr, "ecm: tokenizer blobs missing\n");
        return -1;
    }
    if (tokenizer_load(&g->tok, base + tvocab->offset, tvocab->bytes,
                       base + tmerges->offset, tmerges->bytes, g->vocab) != 0)
        return -1;

    return 0;
}

void gpt2_free(GPT2 *g)
{
    if (!g) return;
    tokenizer_free(&g->tok);
    free(g->lw);
    free(g->k_cache);
    free(g->v_cache);
    free(g->ws_x);
    free(g->ws_tmp);
    free(g->ws_qkv);
    free(g->ws_attn);
    free(g->ws_scores);
    free(g->ws_q);
    free(g->ws_qs);
    if (g->file_data) {
        if (g->file_mapped) munmap(g->file_data, g->file_size);
        else free(g->file_data);
    }
    memset(g, 0, sizeof(*g));
}

/* ------------------------------------------------------------------ */
/* Math helpers                                                         */
/* ------------------------------------------------------------------ */

static inline float gelu_new(float x)
{
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

static void layernorm_rows(float *out, const float *in, const float *gamma,
                           const float *beta, int M, int N, double eps)
{
    for (int m = 0; m < M; m++) {
        const float *row = in + (size_t)m * N;
        float *orow = out + (size_t)m * N;
        double mean = 0.0;
        for (int i = 0; i < N; i++) mean += row[i];
        mean /= N;
        double var = 0.0;
        for (int i = 0; i < N; i++) { double d = row[i] - mean; var += d * d; }
        var /= N;
        double rstd = 1.0 / sqrt(var + eps);
        for (int i = 0; i < N; i++)
            orow[i] = (float)(((double)row[i] - mean) * rstd) * gamma[i] + beta[i];
    }
}

/* ------------------------------------------------------------------ */
/* Linear layer (fp16 or int8)                                          */
/* ------------------------------------------------------------------ */

static void linear(GPT2 *g, const void *w, const float *scale, const float *bias,
                   const float *in, float *out, int M, int K, int N)
{
    if (g->precision == 1) {
        gemm_fp16_fp32(out, in, (const uint16_t *)w, M, K, N);
        if (bias) {
            for (int m = 0; m < M; m++)
                for (int n = 0; n < N; n++)
                    out[(size_t)m * N + n] += bias[n];
        }
    } else {
        int8_t *qa = g->ws_q;
        float *qs = g->ws_qs;
        size_t mcap = (size_t)g->n_ctx > (size_t)g->batch_cap ? (size_t)g->n_ctx : (size_t)g->batch_cap;
        if ((size_t)M * K > mcap * (size_t)(4 * g->n_embd)) {
            fprintf(stderr, "linear: workspace too small\n");
            exit(1);
        }
        for (int m = 0; m < M; m++) {
            const float *row = in + (size_t)m * K;
            float maxa = 0.0f;
            for (int k = 0; k < K; k++) {
                float a = fabsf(row[k]);
                if (a > maxa) maxa = a;
            }
            float s = maxa > 0.0f ? maxa / 127.0f : 1.0f;
            qs[m] = s;
            for (int k = 0; k < K; k++) {
                int32_t q = (int32_t)lrintf(row[k] / s);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                qa[(size_t)m * K + k] = (int8_t)q;
            }
        }
        gemm_int8(out, qa, qs, (const int8_t *)w, scale, bias, M, K, N);
    }
}

/* ------------------------------------------------------------------ */
/* Forward                                                             */
/* ------------------------------------------------------------------ */

void gpt2_forward(GPT2 *g, const int *tokens, const int *positions,
                  const int *seqs, int M, float *logits)
{
    int E = g->n_embd, H = g->n_head, L = g->n_layer, V = g->vocab;
    int hd = E / H;
    const int C = g->n_ctx;
    double inv_hd = 1.0 / sqrt((double)hd);

    float *x = g->ws_x;
    float *h1 = g->ws_tmp;      /* [M][E] */
    float *qkv = g->ws_qkv;     /* [M][3E] */
    float *at = g->ws_attn;     /* [M][E] */
    float *sc = g->ws_scores;   /* [C] */

    /* embeddings */
    for (int m = 0; m < M; m++) {
        const uint16_t *et = g->wte + (size_t)tokens[m] * E;
        const uint16_t *ep = g->wpe + (size_t)positions[m] * E;
        for (int d = 0; d < E; d++)
            x[(size_t)m * E + d] = fp16_to_f32(et[d]) + fp16_to_f32(ep[d]);
    }

    for (int l = 0; l < L; l++) {
        struct LayerW *W = &g->lw[l];

        /* ln1 + qkv */
        layernorm_rows(h1, x, W->ln1_g, W->ln1_b, M, E, g->eps);
        linear(g, W->attn_w, W->attn_s, W->attn_b, h1, qkv, M, E, 3 * E);

        /* store k/v into caches */
        {
            size_t lstride = (size_t)g->batch_cap * C * E;
            for (int m = 0; m < M; m++) {
                int slot = seqs ? seqs[m] : m;
                if (slot >= g->batch_cap) slot = g->batch_cap - 1;
                int pos = positions[m];
                const float *kk = qkv + (size_t)m * (3 * E) + E;
                const float *vv = qkv + (size_t)m * (3 * E) + 2 * E;
                uint16_t *krow = g->k_cache + lstride * (size_t)l + (size_t)slot * (C * E) + (size_t)pos * E;
                uint16_t *vrow = g->v_cache + lstride * (size_t)l + (size_t)slot * (C * E) + (size_t)pos * E;
                for (int d = 0; d < E; d++) {
                    krow[d] = f32_to_fp16(kk[d]);
                    vrow[d] = f32_to_fp16(vv[d]);
                }
            }
        }

        /* attention */
        for (int h = 0; h < H; h++) {
            for (int m = 0; m < M; m++) {
                int slot = seqs ? seqs[m] : m;
                if (slot >= g->batch_cap) slot = g->batch_cap - 1;
                int pos = positions[m];
                const float *qrow = qkv + (size_t)m * (3 * E) + (size_t)h * hd;
                const uint16_t *kbase = g->k_cache + ((size_t)l * g->batch_cap + (size_t)slot) * C * E + (size_t)h * hd;
                const uint16_t *vbase = g->v_cache + ((size_t)l * g->batch_cap + (size_t)slot) * C * E + (size_t)h * hd;
                int npos = pos + 1;
                float maxs = -FLT_MAX;
                for (int p = 0; p < npos; p++) {
                    const uint16_t *kp = kbase + (size_t)p * E;
                    float s = 0.0f;
                    for (int d = 0; d < hd; d++) s += qrow[d] * fp16_to_f32(kp[d]);
                    s = (float)(s * inv_hd);
                    sc[p] = s;
                    if (s > maxs) maxs = s;
                }
                double sume = 0.0;
                for (int p = 0; p < npos; p++) { sc[p] = expf(sc[p] - maxs); sume += sc[p]; }
                double isum = 1.0 / sume;
                for (int p = 0; p < npos; p++) sc[p] = (float)(sc[p] * isum);
                float *orow = at + (size_t)m * E + (size_t)h * hd;
                memset(orow, 0, (size_t)hd * sizeof(float));
                for (int p = 0; p < npos; p++) {
                    const uint16_t *vp = vbase + (size_t)p * E;
                    float w = sc[p];
                    for (int d = 0; d < hd; d++) orow[d] += w * fp16_to_f32(vp[d]);
                }
            }
        }

        /* attn proj + residual */
        linear(g, W->proj_w, W->proj_s, W->proj_b, at, h1, M, E, E);
        for (int m = 0; m < M; m++)
            for (int d = 0; d < E; d++)
                x[(size_t)m * E + d] += h1[(size_t)m * E + d];

        /* mlp */
        layernorm_rows(h1, x, W->ln2_g, W->ln2_b, M, E, g->eps);
        linear(g, W->fc_w, W->fc_s, W->fc_b, h1, qkv, M, E, 4 * E);
        for (int i = 0; i < M * 4 * E; i++) qkv[i] = gelu_new(qkv[i]);
        linear(g, W->fc2_w, W->fc2_s, W->fc2_b, qkv, h1, M, 4 * E, E);
        for (int m = 0; m < M; m++)
            for (int d = 0; d < E; d++)
                x[(size_t)m * E + d] += h1[(size_t)m * E + d];
    }

    /* final layernorm + lm head (tied wte, fp16) */
    layernorm_rows(h1, x, g->lnf_g, g->lnf_b, M, E, g->eps);
    gemm_fp16_fp32(logits, h1, g->wte, M, E, V);
}
