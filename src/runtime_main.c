#define _GNU_SOURCE
#include "edgecore.h"
#include <time.h>
#include <sys/time.h>
#include <sched.h>
#include <unistd.h>

static void json_escape(FILE *o, const char *s);

double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

float peak_rss_gb(void)
{
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return 0.0f;
    char line[256];
    unsigned long kb = 0;
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "VmHWM:", 6)) { sscanf(line + 6, "%lu", &kb); break; }
    }
    fclose(f);
    return (float)kb / 1048576.0f;
}

/* ------------------------------------------------------------------ */
/* CPU affinity                                                         */
/* ------------------------------------------------------------------ */

typedef struct {
    char requested[32];
    int cpus[256];
    int n;
    int applied;            /* 1 if sched_setaffinity succeeded */
    char error[96];
} Affinity;

static void affinity_cpuset_to_list(const cpu_set_t *set, int *cpus, int *n)
{
    *n = 0;
    for (int i = 0; i < CPU_SETSIZE; i++)
        if (CPU_ISSET(i, set) && *n < 256) cpus[(*n)++] = i;
}

static void affinity_current_list(int *cpus, int *n)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    sched_getaffinity(0, sizeof set, &set);
    affinity_cpuset_to_list(&set, cpus, n);
}

/* parse "0" or "0,2,3" into a cpuset; returns 0 on success */
static int affinity_parse_list(const char *s, cpu_set_t *set)
{
    CPU_ZERO(set);
    const char *p = s;
    while (*p) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p) break;
        char *end;
        long cpu = strtol(p, &end, 10);
        if (end == p || cpu < 0 || cpu >= CPU_SETSIZE) return -1;
        CPU_SET((int)cpu, set);
        p = end;
    }
    return 0;
}

/* pick the first logical CPU of each physical core from /proc/cpuinfo */
static int affinity_physical_cpuset(cpu_set_t *set)
{
    CPU_ZERO(set);
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return -1;
    char line[256];
    int cur_proc = -1, cur_phys = -1, cur_core = -1;
    /* map (phys, core) -> representative logical cpu (min processor) */
    typedef struct { int phys, core, cpu; } Core;
    Core cores[256];
    int ncores = 0;
    while (fgets(line, sizeof line, f)) {
        int v;
        if (sscanf(line, "processor : %d", &v) == 1) cur_proc = v;
        else if (sscanf(line, "physical id : %d", &v) == 1) cur_phys = v;
        else if (sscanf(line, "core id : %d", &v) == 1) cur_core = v;
        if (cur_proc >= 0 && cur_phys >= 0 && cur_core >= 0) {
            int found = -1;
            for (int i = 0; i < ncores; i++)
                if (cores[i].phys == cur_phys && cores[i].core == cur_core) { found = i; break; }
            if (found < 0 && ncores < 256) {
                cores[ncores].phys = cur_phys; cores[ncores].core = cur_core;
                cores[ncores].cpu = cur_proc; ncores++;
            } else if (found >= 0 && cur_proc < cores[found].cpu) {
                cores[found].cpu = cur_proc;
            }
        }
    }
    fclose(f);
    for (int i = 0; i < ncores; i++) CPU_SET(cores[i].cpu, set);
    return ncores > 0 ? 0 : -1;
}

/* Apply the requested affinity policy. Fills in `a`. */
static void affinity_apply(const char *policy, Affinity *a)
{
    memset(a, 0, sizeof(*a));
    snprintf(a->requested, sizeof a->requested, "%s", policy);
    cpu_set_t set;
    int want = 0;
    if (!strcmp(policy, "none") || !strcmp(policy, "auto") ||
        !strcmp(policy, "physical")) {
        if (!strcmp(policy, "physical") || !strcmp(policy, "auto")) {
            if (affinity_physical_cpuset(&set) == 0) want = 1;
            else snprintf(a->error, sizeof a->error, "physical core enumeration failed");
        }
    } else {
        if (affinity_parse_list(policy, &set) == 0) want = 1;
        else snprintf(a->error, sizeof a->error, "bad cpu list");
    }
    if (want) {
        if (sched_setaffinity(0, sizeof set, &set) != 0) {
            snprintf(a->error, sizeof a->error, "sched_setaffinity failed");
        } else {
            a->applied = 1;
        }
    }
    affinity_current_list(a->cpus, &a->n);
}

static void emit_affinity(FILE *o, const Affinity *a)
{
    fprintf(o, "  \"affinity_requested\": "); json_escape(o, a->requested);
    fprintf(o, ",\n  \"affinity_applied\": %s,\n", a->applied ? "true" : "false");
    fprintf(o, "  \"affinity_cpus\": [");
    for (int i = 0; i < a->n; i++) fprintf(o, "%s%d", i ? "," : "", a->cpus[i]);
    fprintf(o, "]");
}

static void json_escape(FILE *o, const char *s)
{
    fputc('"', o);
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        switch (c) {
        case '"': fputs("\\\"", o); break;
        case '\\': fputs("\\\\", o); break;
        case '\n': fputs("\\n", o); break;
        case '\t': fputs("\\t", o); break;
        case '\r': fputs("\\r", o); break;
        default:
            if (c < 0x20) fprintf(o, "\\u%04x", c);
            else fputc(c, o);
        }
    }
    fputc('"', o);
}

typedef struct {
    const char *model, *mode, *precision, *prompt, *tokens_file, *output, *affinity;
    const char *logits_bin;
    int threads, batch, n_tokens, warmup, iters, seed, top_k, logits_limit, no_print;
    float temp;
} Args;

static void usage(const char *prog)
{
    fprintf(stderr,
        "usage: %s --model FILE --mode {generate|logits|bench} [options]\n"
        "  --precision fp16|int8|auto (default auto; derived from .ecm weight dtype)\n"
        "  --threads N            (default 1)\n"
        "  --batch N              (default 1)\n"
        "  --prompt TEXT          prompt for generate/logits\n"
        "  --tokens-file FILE     token ids (whitespace separated) to use as prompt\n"
        "  --n-tokens N           tokens to generate (default 32)\n"
        "  --temp T               sampling temperature (default 0.0 = greedy)\n"
        "  --top-k N              top-k sampling (default 0 = off)\n"
        "  --seed N               rng seed\n"
        "  --warmup N / --iters N bench warmup / iterations\n"
        "  --logits-limit N       max prompt tokens for logits mode (default 1024)\n"
        "  --affinity auto|none|physical|0[,2,..]  CPU affinity (default none)\n"
        "  --logits-bin FILE      write raw logits (n, vocab, tokens, logits f32)\n"
        "  --output FILE          JSON output file (default stdout)\n"
        "  --no-print             suppress generated text on stdout\n", prog);
}

static const char *arg_get(int argc, char **argv, int *i, const char *name)
{
    if (*i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", name);
        exit(2);
    }
    (*i)++;
    return argv[*i];
}

static int arg_is(const char *a, const char *b) { return !strcmp(a, b); }

/* simple LCG for sampling */
static unsigned long rng_state;
static unsigned rng_next(void)
{
    rng_state = rng_state * 6364136223846793005UL + 1442695040888963407UL;
    return (unsigned)(rng_state >> 33);
}

static int sample_token(const float *logits, int vocab, float temp, int top_k)
{
    if (temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    /* top-k softmax */
    int *idx = malloc((size_t)vocab * sizeof(int));
    for (int i = 0; i < vocab; i++) idx[i] = i;
    /* partial selection: find top_k via simple sort of top region (vocab small enough) */
    for (int i = 0; i < top_k; i++) {
        int best = i;
        for (int j = i + 1; j < vocab; j++) if (logits[idx[j]] > logits[idx[best]]) best = j;
        int t = idx[i]; idx[i] = idx[best]; idx[best] = t;
    }
    int n = top_k > 0 && top_k < vocab ? top_k : vocab;
    float maxl = logits[idx[0]];
    double sum = 0.0;
    float *p = malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float e = expf((logits[idx[i]] - maxl) / temp);
        p[i] = e;
        sum += e;
    }
    double r = (double)rng_next() / (double)0xffffffffu;
    double acc = 0.0;
    int pick = idx[n - 1];
    for (int i = 0; i < n; i++) {
        acc += p[i] / sum;
        if (r < acc) { pick = idx[i]; break; }
    }
    free(p);
    free(idx);
    return pick;
}

static int read_tokens_file(const char *path, int **out, int *out_n)
{
    FILE *f = fopen(path, "r");
    if (!f) { perror("tokens file"); return -1; }
    int cap = 4096, n = 0;
    int *toks = malloc((size_t)cap * sizeof(int));
    int v;
    while (fscanf(f, "%d", &v) == 1) {
        if (n == cap) { cap *= 2; toks = realloc(toks, (size_t)cap * sizeof(int)); }
        toks[n++] = v;
    }
    fclose(f);
    *out = toks;
    *out_n = n;
    return 0;
}

/* Load a prompt from --tokens-file if given, otherwise tokenize --prompt. */
static int load_prompt(GPT2 *g, const Args *a, int *toks, int cap, int *out_n)
{
    if (a->tokens_file) {
        int *tf = NULL, tn = 0;
        if (read_tokens_file(a->tokens_file, &tf, &tn)) return -1;
        if (tn > cap) tn = cap;
        memcpy(toks, tf, (size_t)tn * sizeof(int));
        free(tf);
        *out_n = tn;
        return 0;
    }
    int n = tokenizer_encode(&g->tok, a->prompt, toks, cap);
    if (n < 0) return -1;
    *out_n = n;
    return 0;
}

/* shared: run prefill + generation; returns stats */
typedef struct {
    double ttft_ms, tpot_avg_ms, p50, p95, p99, tokens_per_sec;
    int n_prompt, n_gen;
    int *ids;   /* combined token ids (prompt + generated) for the first sequence */
    int n_ids;
} GenStats;

static void gen_and_measure(GPT2 *g, int *prompt, int nprompt, int n_gen,
                            int batch, float temp, int top_k, double *tok_ms,
                            int *out_ids, int *out_n, double *ttft_ms)
{
    int V = g->vocab;
    int cap = nprompt + n_gen + 8;
    int *ids = malloc((size_t)cap * sizeof(int));
    memcpy(ids, prompt, (size_t)nprompt * sizeof(int));
    int n_ids = nprompt;

    int *ppos = malloc((size_t)nprompt * sizeof(int));
    int *pseq = malloc((size_t)nprompt * sizeof(int));
    for (int i = 0; i < nprompt; i++) { ppos[i] = i; pseq[i] = 0; }

    int *positions = malloc((size_t)batch * sizeof(int));
    int *cur_tokens = malloc((size_t)batch * sizeof(int));
    int logits_M = nprompt > batch ? nprompt : batch;
    float *logits = malloc((size_t)logits_M * V * sizeof(float));

    /* reset KV: zero caches */
    size_t kv = (size_t)g->n_layer * (size_t)g->batch_cap * (size_t)g->n_ctx * (size_t)g->n_embd;
    memset(g->k_cache, 0, kv * sizeof(uint16_t));
    memset(g->v_cache, 0, kv * sizeof(uint16_t));

    double t0 = now_sec();
    gpt2_forward(g, ids, ppos, pseq, nprompt, logits);
    double t1 = now_sec();
    *ttft_ms = (t1 - t0) * 1000.0;

    int npos = nprompt;             /* absolute position of the next generated token */
    int n_gen_done = 0;
    if (batch > 1) {
        /* prefill remaining batch sequences with same prompt for throughput bench */
        for (int b = 1; b < batch; b++) {
            for (int i = 0; i < nprompt; i++) pseq[i] = b;
            gpt2_forward(g, ids, ppos, pseq, nprompt, logits);
        }
        for (int b = 0; b < batch; b++) cur_tokens[b] = ids[n_ids - 1];
    } else {
        cur_tokens[0] = ids[n_ids - 1];
    }

    while (n_gen_done < n_gen) {
        for (int b = 0; b < batch; b++) positions[b] = npos;
        double s0 = now_sec();
        gpt2_forward(g, cur_tokens, positions, NULL, batch, logits);
        double s1 = now_sec();
        double dt = (s1 - s0) * 1000.0;
        tok_ms[n_gen_done] = dt;
        for (int b = 0; b < batch; b++) {
            int tok = sample_token(logits + (size_t)b * V, V, temp, top_k);
            cur_tokens[b] = tok;
            if (b == 0) {
                if (n_ids < cap) ids[n_ids++] = tok;
            }
        }
        n_gen_done++;
        npos++;
    }

    *out_n = n_ids;
    memcpy(out_ids, ids, (size_t)n_ids * sizeof(int));
    free(ids);
    free(ppos);
    free(pseq);
    free(positions);
    free(cur_tokens);
    free(logits);
}

static int mode_generate(GPT2 *g, const Args *a, const Affinity *aff)
{
    int toks[4096];
    int n;
    if (load_prompt(g, a, toks, 4096, &n)) { fprintf(stderr, "prompt load failed\n"); return 1; }
    if (n < 1) { fprintf(stderr, "empty prompt\n"); return 1; }
    if (n > g->n_ctx - 8) n = g->n_ctx - 8;

    int cap = n + a->n_tokens + 8;
    int *ids = malloc((size_t)cap * sizeof(int));
    double *tok_ms = malloc((size_t)a->n_tokens * sizeof(double));
    double ttft = 0;
    int out_n = 0;
    gen_and_measure(g, toks, n, a->n_tokens, a->batch, a->temp, a->top_k,
                    tok_ms, ids, &out_n, &ttft);

    /* stats */
    double sum = 0;
    for (int i = 0; i < a->n_tokens; i++) sum += tok_ms[i];
    double tpot = a->n_tokens ? sum / a->n_tokens : 0;
    double total_s = (sum + ttft) / 1000.0;
    double tps = total_s > 0 ? (double)a->n_tokens / total_s : 0;
    /* p50/p95/p99 */
    double *srt = malloc((size_t)a->n_tokens * sizeof(double));
    memcpy(srt, tok_ms, (size_t)a->n_tokens * sizeof(double));
    for (int i = 0; i < a->n_tokens; i++)
        for (int j = i + 1; j < a->n_tokens; j++)
            if (srt[j] < srt[i]) { double t = srt[i]; srt[i] = srt[j]; srt[j] = t; }
    double p50 = srt[a->n_tokens ? (a->n_tokens - 1) / 2 : 0];
    double p95 = srt[a->n_tokens ? (int)(a->n_tokens * 0.95) : 0];
    double p99 = srt[a->n_tokens ? (int)(a->n_tokens * 0.99) : 0];
    if (a->n_tokens > 1) { if (p95 > srt[a->n_tokens - 1]) p95 = srt[a->n_tokens - 1]; if (p99 > srt[a->n_tokens - 1]) p99 = srt[a->n_tokens - 1]; }

    char *text = NULL;
    if (!a->no_print) text = tokenizer_decode(&g->tok, ids, out_n);

    FILE *o = stdout;
    if (a->output) o = fopen(a->output, "w");
    fprintf(o, "{\n");
    fprintf(o, "  \"mode\": \"generate\",\n");
    fprintf(o, "  \"model\": "); json_escape(o, g->model_name); fprintf(o, ",\n");
    fprintf(o, "  \"precision\": \"%s\",\n", a->precision);
    fprintf(o, "  \"threads\": %d,\n", a->threads);
    fprintf(o, "  \"batch\": %d,\n", a->batch);
    emit_affinity(o, aff);
    fprintf(o, ",\n  \"n_prompt_tokens\": %d,\n", n);
    fprintf(o, "  \"n_generated_tokens\": %d,\n", a->n_tokens);
    fprintf(o, "  \"ttft_ms\": %.3f,\n", ttft);
    fprintf(o, "  \"tpot_avg_ms\": %.3f,\n", tpot);
    fprintf(o, "  \"tokens_per_sec\": %.3f,\n", tps);
    fprintf(o, "  \"p50_ms\": %.3f,\n", p50);
    fprintf(o, "  \"p95_ms\": %.3f,\n", p95);
    fprintf(o, "  \"p99_ms\": %.3f,\n", p99);
    fprintf(o, "  \"peak_rss_gb\": %.3f", peak_rss_gb());
    if (text) { fprintf(o, ",\n  \"text\": "); json_escape(o, text); }
    fprintf(o, "\n}\n");
    if (a->output) fclose(o);

    if (text) {
        printf("%s\n", text);
        free(text);
    }
    free(ids);
    free(tok_ms);
    free(srt);
    return 0;
}

static int mode_logits(GPT2 *g, const Args *a, const Affinity *aff)
{
    int toks[4096];
    int n;
    if (load_prompt(g, a, toks, 4096, &n)) { fprintf(stderr, "prompt load failed\n"); return 1; }
    if (n < 1) return 1;
    if (n > a->logits_limit) n = a->logits_limit;

    int V = g->vocab;
    int *positions = malloc((size_t)n * sizeof(int));
    float *logits = malloc((size_t)n * V * sizeof(float));
    int *seqs = malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) { positions[i] = i; seqs[i] = 0; }
    size_t kv = (size_t)g->n_layer * (size_t)g->batch_cap * (size_t)g->n_ctx * (size_t)g->n_embd;
    memset(g->k_cache, 0, kv * sizeof(uint16_t));
    memset(g->v_cache, 0, kv * sizeof(uint16_t));
    gpt2_forward(g, toks, positions, seqs, n, logits);

    if (a->logits_bin) {
        /* raw binary dump: int32 n, int32 V, tokens[n], logits[n*V] f32 */
        FILE *b = fopen(a->logits_bin, "wb");
        if (!b) { perror("logits bin"); free(positions); free(logits); free(seqs); return 1; }
        int32_t hdr[2] = { n, V };
        fwrite(hdr, sizeof(int32_t), 2, b);
        fwrite(toks, sizeof(int32_t), (size_t)n, b);
        for (int i = 0; i < n; i++)
            fwrite(logits + (size_t)i * V, sizeof(float), (size_t)V, b);
        fclose(b);
    }

    FILE *o = stdout;
    if (a->output) o = fopen(a->output, "w");
    fprintf(o, "{\n");
    fprintf(o, "  \"model\": "); json_escape(o, g->model_name); fprintf(o, ",\n");
    fprintf(o, "  \"precision\": \"%s\",\n", a->precision);
    fprintf(o, "  \"threads\": %d,\n", a->threads);
    emit_affinity(o, aff);
    fprintf(o, ",\n  \"n_tokens\": %d,\n", n);
    fprintf(o, "  \"tokens\": [");
    for (int i = 0; i < n; i++) fprintf(o, "%s%d", i ? "," : "", toks[i]);
    fprintf(o, "],\n  \"logits\": [\n");
    for (int i = 0; i < n; i++) {
        fprintf(o, "    [");
        for (int j = 0; j < V; j++) {
            float v = logits[(size_t)i * V + j];
            if (j) fputc(',', o);
            if (isfinite(v)) fprintf(o, "%.6g", v);
            else fputs("null", o);
        }
        fprintf(o, "]%s\n", i + 1 < n ? "," : "");
    }
    fprintf(o, "  ]\n}\n");
    if (a->output) fclose(o);

    free(positions);
    free(logits);
    free(seqs);
    return 0;
}

static int mode_bench(GPT2 *g, const Args *a, const Affinity *aff)
{
    int toks[4096];
    int n;
    if (load_prompt(g, a, toks, 4096, &n)) { fprintf(stderr, "prompt load failed\n"); return 1; }
    if (n < 1) { fprintf(stderr, "empty prompt\n"); return 1; }
    if (n > g->n_ctx - 8) n = g->n_ctx - 8;

    int total_gen = a->batch * a->n_tokens;
    double *tok_ms = malloc((size_t)a->n_tokens * sizeof(double));
    int *ids = malloc((size_t)(n + a->n_tokens + 8) * sizeof(int));

    for (int w = 0; w < a->warmup; w++) {
        double ttft = 0; int out_n = 0;
        gen_and_measure(g, toks, n, a->n_tokens, a->batch, 0, 0, tok_ms, ids, &out_n, &ttft);
    }

    int iters = a->iters > 0 ? a->iters : 1;
    double *all_ms = malloc((size_t)(total_gen * iters) * sizeof(double));
    double *ttfts = malloc((size_t)iters * sizeof(double));
    int idx = 0;
    for (int it = 0; it < iters; it++) {
        double ttft = 0; int out_n = 0;
        gen_and_measure(g, toks, n, a->n_tokens, a->batch, 0, 0, tok_ms, ids, &out_n, &ttft);
        ttfts[it] = ttft;
        /* each timing sample covers `batch` tokens; expand to per-token samples */
        for (int i = 0; i < a->n_tokens; i++)
            for (int b = 0; b < a->batch; b++) all_ms[idx++] = tok_ms[i];
    }

    /* per-token latencies across all steps */
    int N = idx;
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            if (all_ms[j] < all_ms[i]) { double t = all_ms[i]; all_ms[i] = all_ms[j]; all_ms[j] = t; }
    double sum = 0;
    for (int i = 0; i < N; i++) sum += all_ms[i];
    double avg = N ? sum / N : 0;
    double p50 = N ? all_ms[(N - 1) / 2] : 0;
    double p95 = N ? all_ms[(int)(N * 0.95)] : 0;
    double p99 = N ? all_ms[(int)(N * 0.99)] : 0;
    if (N > 1) { if (p95 > all_ms[N - 1]) p95 = all_ms[N - 1]; if (p99 > all_ms[N - 1]) p99 = all_ms[N - 1]; }
    double gen_s = sum / 1000.0;

    /* TTFT percentiles across iterations */
    double *st = malloc((size_t)iters * sizeof(double));
    memcpy(st, ttfts, (size_t)iters * sizeof(double));
    for (int i = 0; i < iters; i++)
        for (int j = i + 1; j < iters; j++)
            if (st[j] < st[i]) { double t = st[i]; st[i] = st[j]; st[j] = t; }
    double ttft_avg = 0;
    for (int i = 0; i < iters; i++) ttft_avg += ttfts[i] / (double)iters;
    double ttft_p50 = st[iters ? (iters - 1) / 2 : 0];
    double ttft_p95 = st[iters ? (int)(iters * 0.95) : 0];
    double ttft_p99 = st[iters ? (int)(iters * 0.99) : 0];
    if (iters > 1) { if (ttft_p95 > st[iters - 1]) ttft_p95 = st[iters - 1]; if (ttft_p99 > st[iters - 1]) ttft_p99 = st[iters - 1]; }
    double tps = gen_s > 0 ? (double)N / gen_s : 0;

    FILE *o = stdout;
    if (a->output) o = fopen(a->output, "w");
    fprintf(o, "{\n");
    fprintf(o, "  \"mode\": \"bench\",\n");
    fprintf(o, "  \"model\": "); json_escape(o, g->model_name); fprintf(o, ",\n");
    fprintf(o, "  \"precision\": \"%s\",\n", a->precision);
    fprintf(o, "  \"threads\": %d,\n", a->threads);
    fprintf(o, "  \"batch\": %d,\n", a->batch);
    emit_affinity(o, aff);
    fprintf(o, ",\n  \"n_prompt_tokens\": %d,\n", n);
    fprintf(o, "  \"n_generated_tokens\": %d,\n", total_gen);
    fprintf(o, "  \"iterations\": %d,\n", iters);
    fprintf(o, "  \"avg_token_ms\": %.3f,\n", avg);
    fprintf(o, "  \"p50_ms\": %.3f,\n", p50);
    fprintf(o, "  \"p95_ms\": %.3f,\n", p95);
    fprintf(o, "  \"p99_ms\": %.3f,\n", p99);
    fprintf(o, "  \"ttft_avg_ms\": %.3f,\n", ttft_avg);
    fprintf(o, "  \"ttft_p50_ms\": %.3f,\n", ttft_p50);
    fprintf(o, "  \"ttft_p95_ms\": %.3f,\n", ttft_p95);
    fprintf(o, "  \"ttft_p99_ms\": %.3f,\n", ttft_p99);
    fprintf(o, "  \"tokens_per_sec\": %.3f,\n", tps);
    fprintf(o, "  \"total_generation_sec\": %.3f,\n", gen_s);
    fprintf(o, "  \"peak_rss_gb\": %.3f\n", peak_rss_gb());
    fprintf(o, "}\n");
    if (a->output) fclose(o);

    free(tok_ms);
    free(ids);
    free(all_ms);
    free(ttfts);
    free(st);
    return 0;
}

int main(int argc, char **argv)
{
    Args a = {0};
    a.mode = "generate";
    a.precision = "auto";
    a.affinity = "none";
    a.threads = 1;
    a.batch = 1;
    a.n_tokens = 32;
    a.warmup = 1;
    a.iters = 1;
    a.logits_limit = 1024;
    a.temp = 0.0f;
    a.top_k = 0;
    a.seed = 42;
    a.prompt = "The quick brown fox jumps over the lazy dog";

    if (argc < 2) { usage(argv[0]); return 2; }
    for (int i = 1; i < argc; i++) {
        if (arg_is(argv[i], "--model")) a.model = arg_get(argc, argv, &i, "--model");
        else if (arg_is(argv[i], "--mode")) a.mode = arg_get(argc, argv, &i, "--mode");
        else if (arg_is(argv[i], "--precision")) a.precision = arg_get(argc, argv, &i, "--precision");
        else if (arg_is(argv[i], "--threads")) a.threads = atoi(arg_get(argc, argv, &i, "--threads"));
        else if (arg_is(argv[i], "--batch")) a.batch = atoi(arg_get(argc, argv, &i, "--batch"));
        else if (arg_is(argv[i], "--prompt")) a.prompt = arg_get(argc, argv, &i, "--prompt");
        else if (arg_is(argv[i], "--tokens-file")) a.tokens_file = arg_get(argc, argv, &i, "--tokens-file");
        else if (arg_is(argv[i], "--n-tokens")) a.n_tokens = atoi(arg_get(argc, argv, &i, "--n-tokens"));
        else if (arg_is(argv[i], "--temp")) a.temp = atof(arg_get(argc, argv, &i, "--temp"));
        else if (arg_is(argv[i], "--top-k")) a.top_k = atoi(arg_get(argc, argv, &i, "--top-k"));
        else if (arg_is(argv[i], "--seed")) a.seed = atoi(arg_get(argc, argv, &i, "--seed"));
        else if (arg_is(argv[i], "--warmup")) a.warmup = atoi(arg_get(argc, argv, &i, "--warmup"));
        else if (arg_is(argv[i], "--iters")) a.iters = atoi(arg_get(argc, argv, &i, "--iters"));
        else if (arg_is(argv[i], "--logits-limit")) a.logits_limit = atoi(arg_get(argc, argv, &i, "--logits-limit"));
        else if (arg_is(argv[i], "--output")) a.output = arg_get(argc, argv, &i, "--output");
        else if (arg_is(argv[i], "--affinity")) a.affinity = arg_get(argc, argv, &i, "--affinity");
        else if (arg_is(argv[i], "--logits-bin")) a.logits_bin = arg_get(argc, argv, &i, "--logits-bin");
        else if (arg_is(argv[i], "--no-print")) a.no_print = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 2; }
    }
    if (!a.model) { usage(argv[0]); return 2; }

    rng_state = (unsigned long)a.seed;
    omp_set_num_threads(a.threads > 0 ? a.threads : 1);

    Affinity aff;
    affinity_apply(a.affinity, &aff);

    int want_precision = -1;
    if (!strcmp(a.precision, "fp16")) want_precision = 1;
    else if (!strcmp(a.precision, "int8")) want_precision = 0;
    else if (strcmp(a.precision, "auto")) {
        fprintf(stderr, "unknown precision: %s (use fp16|int8|auto)\n", a.precision);
        return 2;
    }

    GPT2 g;
    if (gpt2_load(&g, a.model, a.batch) != 0) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    if (want_precision >= 0 && want_precision != g.precision) {
        fprintf(stderr, "model file uses %s weights, but --precision %s was requested\n",
                g.precision ? "fp16" : "int8", a.precision);
        gpt2_free(&g);
        return 2;
    }
    a.precision = g.precision ? "fp16" : "int8";

    int rc = 0;
    if (!strcmp(a.mode, "generate")) rc = mode_generate(&g, &a, &aff);
    else if (!strcmp(a.mode, "logits")) rc = mode_logits(&g, &a, &aff);
    else if (!strcmp(a.mode, "bench")) rc = mode_bench(&g, &a, &aff);
    else { fprintf(stderr, "unknown mode: %s\n", a.mode); rc = 2; }

    gpt2_free(&g);
    return rc;
}
