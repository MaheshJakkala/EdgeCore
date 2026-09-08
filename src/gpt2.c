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
    case 4: return 2; /* bf16 */
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
        else if (!strcmp(key, "n_kv_head")) meta->n_kv_head = atoi(val);
        else if (!strcmp(key, "intermediate")) meta->intermediate = atoi(val);
        else if (!strcmp(key, "n_embd")) meta->n_embd = atoi(val);
        else if (!strcmp(key, "n_ctx")) meta->n_ctx = atoi(val);
        else if (!strcmp(key, "vocab")) meta->vocab = atoi(val);
        else if (!strcmp(key, "n_params")) meta->n_params = (size_t)strtoull(val, NULL, 10);
        else if (!strcmp(key, "eps")) meta->eps = atof(val);
        else if (!strcmp(key, "rope_theta")) meta->rope_theta = atof(val);
        else if (!strcmp(key, "rope_type")) {
            meta->rope_type = !strcmp(val, "yarn") ? ROPE_YARN : ROPE_DEFAULT;
        }         else if (!strcmp(key, "yarn_factor")) meta->yarn_factor = atof(val);
        else if (!strcmp(key, "yarn_attention_factor")) meta->yarn_attention_factor = atof(val);
        else if (!strcmp(key, "yarn_orig_max")) meta->yarn_orig_max = atof(val);
        else if (!strcmp(key, "sliding_window")) meta->sliding_window = atoi(val);
        else if (!strcmp(key, "tie_word_embeddings")) meta->tie_word_embeddings = atoi(val);
        else if (!strcmp(key, "eos_token_id")) meta->eos_token_id = atoi(val);
        else if (!strcmp(key, "arch")) {
            if (!strcmp(val, "qwen2") || !strcmp(val, "olmo3")) meta->arch = ARCH_QWEN2;
            else meta->arch = ARCH_GPT2;
        }
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
        else if (!strcmp(dtype_s, "bf16")) t->dtype = 4;
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

static void set_qwen2_defaults(GPT2 *g)
{
    if (g->n_kv_head <= 0) g->n_kv_head = g->n_head;
    g->head_dim = g->n_embd / (g->n_head > 0 ? g->n_head : 1);
    g->n_kv_embd = g->n_kv_head * g->head_dim;
    if (g->intermediate <= 0) g->intermediate = 4 * g->n_embd;
    if (g->rope_theta <= 0) g->rope_theta = 10000.0;
    if (g->yarn_orig_max <= 0) g->yarn_orig_max = g->n_ctx;
    if (g->yarn_factor <= 0) g->yarn_factor = 1.0;
    if (g->yarn_attention_factor <= 0) g->yarn_attention_factor = 1.0;
}

int gpt2_load(GPT2 *g, const char *path, int batch_cap, int max_ctx)
{
    memset(g, 0, sizeof(*g));
    g->batch_cap = batch_cap < 1 ? 1 : batch_cap;
    g->max_ctx = max_ctx > 0 ? max_ctx : 0;
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
    const EcmTensor *lnf_g = find_tensor(tens, n, "ln_f.g");
    if (!wte || !lnf_g) {
        fprintf(stderr, "ecm: missing core tensors (wte, ln_f.g)\n");
        return -1;
    }
    g->wte = (const uint16_t *)(base + wte->offset);
    g->lnf_g = (const float *)(base + lnf_g->offset);
    g->lnf_b = NULL;
    g->wpe = NULL;

    /* precision is determined by the file: the linear weights are stored
     * either fp16, bf16 (native bf16 artifact) or int8 (per-channel quant). */
    const char *lin0 = g->arch == ARCH_QWEN2 ? "blocks.0.q.w" : "blocks.0.attn.w";
    const EcmTensor *t0 = find_tensor(tens, n, lin0);
    if (!t0) {
        fprintf(stderr, "ecm: missing %s\n", lin0);
        return -1;
    }
    if (t0->dtype == 1) {
        g->precision = 1;   /* fp16 */
    } else if (t0->dtype == 2) {
        g->precision = 0;   /* int8 */
    } else if (t0->dtype == 4) {
        g->precision = 2;   /* bf16 */
    } else {
        fprintf(stderr, "ecm: unexpected linear weight dtype\n");
        return -1;
    }

    if (g->arch == ARCH_QWEN2) set_qwen2_defaults(g);

    if (g->arch == ARCH_GPT2) {
        const EcmTensor *wpe = find_tensor(tens, n, "wpe");
        const EcmTensor *lnf_b = find_tensor(tens, n, "ln_f.b");
        if (!wpe || !lnf_b) {
            fprintf(stderr, "ecm: missing wpe / ln_f.b\n");
            return -1;
        }
        g->wpe = (const uint16_t *)(base + wpe->offset);
        g->lnf_b = (const float *)(base + lnf_b->offset);
        g->n_kv_embd = g->n_embd;
        g->n_kv_head = g->n_head;
        g->intermediate = 4 * g->n_embd;
        g->head_dim = g->n_embd / g->n_head;
    } else {
        const EcmTensor *lmh = find_tensor(tens, n, "lm_head.w");
        if (lmh && !g->tie_word_embeddings)
            g->lm_head = (const uint16_t *)(base + lmh->offset);
        if (g->tie_word_embeddings || !lmh)
            g->tie_word_embeddings = 1;
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

        if (g->arch == ARCH_QWEN2) {
            struct { const char *suff; const void **w; const float **s; const float **b; } p[7] = {
                {"q", &W->q_w, &W->q_s, &W->q_b},
                {"k", &W->k_w, &W->k_s, &W->k_b},
                {"v", &W->v_w, &W->v_s, &W->v_b},
                {"o", &W->o_w, &W->o_s, &W->o_b},
                {"gate", &W->gate_w, &W->gate_s, &W->gate_b},
                {"up", &W->up_w, &W->up_s, &W->up_b},
                {"down", &W->down_w, &W->down_s, &W->down_b},
            };
            for (int s = 0; s < 7; s++) {
                snprintf(nm, sizeof nm, "%s.%s.w", prefix, p[s].suff);
                const EcmTensor *tw = find_tensor(tens, n, nm);
                snprintf(nm, sizeof nm, "%s.%s.s", prefix, p[s].suff);
                const EcmTensor *ts = find_tensor(tens, n, nm);
                snprintf(nm, sizeof nm, "%s.%s.b", prefix, p[s].suff);
                const EcmTensor *tb = find_tensor(tens, n, nm);
                if (!tw) {
                    fprintf(stderr, "ecm: missing %s\n", nm);
                    return -1;
                }
                *p[s].w = base + tw->offset;
                *p[s].s = (g->precision == 0 && ts) ? (const float *)(base + ts->offset) : NULL;
                *p[s].b = tb ? (const float *)(base + tb->offset) : NULL;
            }
        } else {
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
    }

    int eff_ctx = (g->max_ctx > 0 && g->max_ctx < g->n_ctx) ? g->max_ctx : g->n_ctx;

    /* KV cache: [n_layer][batch][eff_ctx][n_kv_embd] fp16 */
    size_t kv = (size_t)g->n_layer * (size_t)g->batch_cap * (size_t)eff_ctx *
                (size_t)g->n_kv_embd;
    g->k_cache = calloc(kv, sizeof(uint16_t));
    g->v_cache = calloc(kv, sizeof(uint16_t));
    if (!g->k_cache || !g->v_cache) return -1;

    /* workspace: sized for the largest M = max(eff_ctx, batch_cap) so that
     * prefill (M up to eff_ctx) and batched decode (M up to batch_cap) fit. */
    size_t E = (size_t)g->n_embd;
    size_t B = (size_t)g->batch_cap;
    size_t Mmax = (size_t)eff_ctx > B ? (size_t)eff_ctx : B;
    size_t qkv_cols;
    if (g->arch == ARCH_QWEN2) {
        /* qkv output uses q [M,E], k [M,n_kv_embd], v [M,n_kv_embd]; MLP
         * gate/up outputs [M,intermediate] each (computed into ws_qkv). */
        size_t a = 2 * (size_t)g->intermediate;
        size_t b = E + 2 * (size_t)g->n_kv_embd;
        qkv_cols = a > b ? a : b;
    } else {
        /* gpt2: ws_qkv holds qkv [M][3E] AND is reused as the MLP fc
         * output / fc2 input [M][4E], so it must be sized for 4E. */
        qkv_cols = 4 * E;
    }
    g->ws_x = malloc(Mmax * E * sizeof(float));
    g->ws_tmp = malloc(Mmax * (4 * E) * sizeof(float));
    g->ws_qkv = malloc(Mmax * qkv_cols * sizeof(float));
    g->ws_attn = malloc(Mmax * E * sizeof(float));
    g->ws_scores = malloc((size_t)eff_ctx * sizeof(float));
    {
        size_t qmax = (size_t)g->arch == ARCH_QWEN2
            ? (E > (size_t)g->intermediate ? E : (size_t)g->intermediate)
            : 4 * E;
        g->ws_q_cols = qmax;
        g->ws_q = malloc(Mmax * qmax * sizeof(int8_t));
    }
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
                       base + tmerges->offset, tmerges->bytes, g->vocab,
                       g->arch == ARCH_QWEN2 ? 1 : 0) != 0)
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
    } else if (g->precision == 2) {
        gemm_bf16_fp32(out, in, (const uint16_t *)w, M, K, N);
        if (bias) {
            for (int m = 0; m < M; m++)
                for (int n = 0; n < N; n++)
                    out[(size_t)m * N + n] += bias[n];
        }
    } else {
        int8_t *qa = g->ws_q;
        float *qs = g->ws_qs;
        int eff_ctx = (g->max_ctx > 0 && g->max_ctx < g->n_ctx) ? g->max_ctx : g->n_ctx;
        size_t mcap = (size_t)eff_ctx > (size_t)g->batch_cap ? (size_t)eff_ctx : (size_t)g->batch_cap;
        if ((size_t)M * K > mcap * g->ws_q_cols) {
            fprintf(stderr, "linear: workspace too small (M=%d K=%d)\n", M, K);
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
    if (g->arch == ARCH_QWEN2) {
        qwen2_forward(g, tokens, positions, seqs, M, logits);
        return;
    }
    int E = g->n_embd, H = g->n_head, L = g->n_layer, V = g->vocab;
    int hd = E / H;
    const int C = (g->max_ctx > 0 && g->max_ctx < g->n_ctx) ? g->max_ctx : g->n_ctx;
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
        if (g->precision == 2) {
            for (int d = 0; d < E; d++)
                x[(size_t)m * E + d] = bf16_to_f32(et[d]) + bf16_to_f32(ep[d]);
        } else {
            for (int d = 0; d < E; d++)
                x[(size_t)m * E + d] = fp16_to_f32(et[d]) + fp16_to_f32(ep[d]);
        }
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
    if (g->precision == 2)
        gemm_bf16_fp32(logits, h1, g->wte, M, E, V);
    else
        gemm_fp16_fp32(logits, h1, g->wte, M, E, V);
}

/* ------------------------------------------------------------------ */
/* Qwen2 / OLMo decoder (RoPE + RMSNorm + GQA + SwiGLU)                */
/* ------------------------------------------------------------------ */

static inline float qwen_silu(float x)
{
    return x / (1.0f + expf(-x));
}

static void rmsnorm_rows(float *out, const float *in, const float *gamma,
                         int M, int N, double eps)
{
    for (int m = 0; m < M; m++) {
        const float *row = in + (size_t)m * N;
        float *orow = out + (size_t)m * N;
        double ms = 0.0;
        for (int i = 0; i < N; i++) ms += (double)row[i] * row[i];
        double rstd = 1.0 / sqrt(ms / N + eps);
        for (int i = 0; i < N; i++)
            orow[i] = (float)((double)row[i] * rstd) * gamma[i];
    }
}

/* inverse frequencies for one head, matching transformers
 * _compute_yarn_parameters when rope_type=yarn (OLMo / Spark). */
static void qwen_inv_freq(double *inv_freq, int half, const GPT2 *g)
{
    int dim = 2 * half;
    double base = g->rope_theta;
    double inv_base = 1.0 / base;
    for (int i = 0; i < half; i++)
        inv_freq[i] = pow(inv_base, (double)(2 * i) / (double)dim);

    if (g->rope_type != ROPE_YARN) return;

    double factor = g->yarn_factor;
    double max_pos = g->yarn_orig_max;
    double beta_fast = 32.0, beta_slow = 1.0;
    double pi2 = 6.28318530717958647692;
    double logbase = log(base);

    /* find_correction_dim */
    double corr_dim(double rot) {
        return (double)dim * log(max_pos / (rot * pi2)) / (2.0 * logbase);
    }
    double lo = floor(corr_dim(beta_fast));
    double hi = ceil(corr_dim(beta_slow));
    if (lo < 0.0) lo = 0.0;
    if (hi > (double)(dim - 1)) hi = (double)(dim - 1);
    double span = hi - lo;
    if (span < 1.0) span = 1.0;

    /* inv_freq = interp * ramp + extrap * (1 - ramp) */
    for (int i = 0; i < half; i++) {
        double ramp = ((double)i - lo) / span;
        if (ramp < 0.0) ramp = 0.0;
        if (ramp > 1.0) ramp = 1.0;
        double extrap = inv_freq[i];
        double interp = inv_freq[i] / factor;
        inv_freq[i] = interp * ramp + extrap * (1.0 - ramp);
    }
}

static void qwen_rope_cos_sin(const double *inv_freq, int half, int pos,
                              double scale, double *cosv, double *sinv)
{
    for (int i = 0; i < half; i++) {
        double f = (double)pos * inv_freq[i];
        cosv[i] = cos(f) * scale;
        sinv[i] = sin(f) * scale;
    }
}

static void qwen_rope_rotate(float *v, int half,
                             const double *cosv, const double *sinv)
{
    for (int i = 0; i < half; i++) {
        float a = v[i];
        float b = v[i + half];
        v[i] = (float)(a * cosv[i] - b * sinv[i]);
        v[i + half] = (float)(a * sinv[i] + b * cosv[i]);
    }
}

void qwen2_forward(GPT2 *g, const int *tokens, const int *positions,
                   const int *seqs, int M, float *logits)
{
    int E = g->n_embd, H = g->n_head, L = g->n_layer, V = g->vocab;
    int hd = g->head_dim;
    int half = hd / 2;
    int kv_embd = g->n_kv_embd;
    int inter = g->intermediate;
    const int C = (g->max_ctx > 0 && g->max_ctx < g->n_ctx) ? g->max_ctx : g->n_ctx;
    double inv_hd = 1.0 / sqrt((double)hd);
    double attn_scale = g->yarn_attention_factor;

    float *x = g->ws_x;         /* [M][E] residual stream */
    float *h1 = g->ws_tmp;      /* [M][E] (sized [M][4E]) */
    float *qkv = g->ws_qkv;     /* [M][E+2*kv_embd] q/k/v, then [M][2*inter] mlp */
    float *at = g->ws_attn;     /* [M][E] attention output */
    float *sc = g->ws_scores;   /* [C] */

    double inv_freq[1024];
    if (half > 512) half = 512;
    qwen_inv_freq(inv_freq, half, g);
    double cosv[512], sinv[512];

    /* embeddings (no positional embeddings; RoPE is positional) */
    for (int m = 0; m < M; m++) {
        const uint16_t *et = g->wte + (size_t)tokens[m] * E;
        if (g->precision == 2) {
            for (int d = 0; d < E; d++)
                x[(size_t)m * E + d] = bf16_to_f32(et[d]);
        } else {
            for (int d = 0; d < E; d++)
                x[(size_t)m * E + d] = fp16_to_f32(et[d]);
        }
    }

    for (int l = 0; l < L; l++) {
        struct LayerW *W = &g->lw[l];

        /* input RMSNorm */
        rmsnorm_rows(h1, x, W->ln1_g, M, E, g->eps);

        /* q/k/v projections */
        linear(g, W->q_w, W->q_s, NULL, h1, qkv, M, E, E);
        linear(g, W->k_w, W->k_s, NULL, h1, qkv + (size_t)M * E, M, E, kv_embd);
        linear(g, W->v_w, W->v_s, NULL, h1, qkv + (size_t)M * (E + kv_embd), M, E, kv_embd);

        /* RoPE on q and k */
        for (int m = 0; m < M; m++) {
            int pos = positions[m];
            qwen_rope_cos_sin(inv_freq, half, pos, attn_scale, cosv, sinv);
            float *qrow = qkv + (size_t)m * E;
            for (int h = 0; h < H; h++)
                qwen_rope_rotate(qrow + (size_t)h * hd, half, cosv, sinv);
            float *krow = qkv + (size_t)M * E + (size_t)m * kv_embd;
            for (int g2 = 0; g2 < g->n_kv_head; g2++)
                qwen_rope_rotate(krow + (size_t)g2 * hd, half, cosv, sinv);
        }

        /* store k/v into caches */
        {
            size_t lstride = (size_t)g->batch_cap * C * kv_embd;
            for (int m = 0; m < M; m++) {
                int slot = seqs ? seqs[m] : m;
                if (slot >= g->batch_cap) slot = g->batch_cap - 1;
                int pos = positions[m];
                const float *kk = qkv + (size_t)M * E + (size_t)m * kv_embd;
                const float *vv = qkv + (size_t)M * (E + kv_embd) + (size_t)m * kv_embd;
                uint16_t *krow = g->k_cache + lstride * (size_t)l + (size_t)slot * (C * kv_embd) + (size_t)pos * kv_embd;
                uint16_t *vrow = g->v_cache + lstride * (size_t)l + (size_t)slot * (C * kv_embd) + (size_t)pos * kv_embd;
                for (int d = 0; d < kv_embd; d++) {
                    krow[d] = f32_to_fp16(kk[d]);
                    vrow[d] = f32_to_fp16(vv[d]);
                }
            }
        }

        /* grouped-query attention (head h attends to kv head h*n_kv_head/H) */
        for (int h = 0; h < H; h++) {
            int kvh = h * g->n_kv_head / H;
            for (int m = 0; m < M; m++) {
                int slot = seqs ? seqs[m] : m;
                if (slot >= g->batch_cap) slot = g->batch_cap - 1;
                int pos = positions[m];
                const float *qrow = qkv + (size_t)m * E + (size_t)h * hd;
                const uint16_t *kbase = g->k_cache + ((size_t)l * g->batch_cap + (size_t)slot) * (C * kv_embd) + (size_t)kvh * hd;
                const uint16_t *vbase = g->v_cache + ((size_t)l * g->batch_cap + (size_t)slot) * (C * kv_embd) + (size_t)kvh * hd;
                int lo = g->sliding_window > 0 ? (pos - g->sliding_window + 1) : 0;
                if (lo < 0) lo = 0;
                int npos = pos + 1;
                float maxs = -FLT_MAX;
                for (int p = lo; p < npos; p++) {
                    const uint16_t *kp = kbase + (size_t)p * kv_embd;
                    float s = 0.0f;
                    for (int d = 0; d < hd; d++) s += qrow[d] * fp16_to_f32(kp[d]);
                    s = (float)(s * inv_hd);
                    sc[p] = s;
                    if (s > maxs) maxs = s;
                }
                double sume = 0.0;
                for (int p = lo; p < npos; p++) { sc[p] = expf(sc[p] - maxs); sume += sc[p]; }
                double isum = 1.0 / sume;
                for (int p = lo; p < npos; p++) sc[p] = (float)(sc[p] * isum);
                float *orow = at + (size_t)m * E + (size_t)h * hd;
                memset(orow, 0, (size_t)hd * sizeof(float));
                for (int p = lo; p < npos; p++) {
                    const uint16_t *vp = vbase + (size_t)p * kv_embd;
                    float w = sc[p];
                    for (int d = 0; d < hd; d++) orow[d] += w * fp16_to_f32(vp[d]);
                }
            }
        }

        /* output projection + residual (o_proj maps full attention output
         * [M][E] back to the residual width [M][E]; weight stored [E][E]) */
        linear(g, W->o_w, W->o_s, NULL, at, h1, M, E, E);
        for (int m = 0; m < M; m++)
            for (int d = 0; d < E; d++)
                x[(size_t)m * E + d] += h1[(size_t)m * E + d];

        /* SwiGLU MLP: gate/up -> silu(gate)*up -> down, residual */
        rmsnorm_rows(h1, x, W->ln2_g, M, E, g->eps);
        linear(g, W->gate_w, W->gate_s, NULL, h1, qkv, M, E, inter);
        linear(g, W->up_w, W->up_s, NULL, h1, qkv + (size_t)M * inter, M, E, inter);
        for (int i = 0; i < M * inter; i++)
            qkv[i] = qwen_silu(qkv[i]) * qkv[(size_t)M * inter + i];
        linear(g, W->down_w, W->down_s, NULL, qkv, h1, M, inter, E);
        for (int m = 0; m < M; m++)
            for (int d = 0; d < E; d++)
                x[(size_t)m * E + d] += h1[(size_t)m * E + d];
    }

    /* final RMSNorm + lm head */
    rmsnorm_rows(h1, x, g->lnf_g, M, E, g->eps);
    const uint16_t *lm = g->lm_head ? g->lm_head : g->wte;
    if (g->precision == 2)
        gemm_bf16_fp32(logits, h1, lm, M, E, V);
    else
        gemm_fp16_fp32(logits, h1, lm, M, E, V);
}
