#ifndef EDGECORE_H
#define EDGECORE_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define ECM_MAGIC "ECM1\n"

/* Architecture identifiers stored in the .ecm meta `arch` field. */
#define ARCH_GPT2  0   /* GPT-2 (learned positional embeddings, LN, GeLU) */
#define ARCH_QWEN2 1   /* Qwen2 / OLMo-style decoder (RoPE, RMSNorm, SwiGLU, GQA) */

/* Rope scaling variants (meta `rope_type`). */
#define ROPE_DEFAULT 0
#define ROPE_YARN    1

/* ------------------------------------------------------------------ */
/* fp16 conversion                                                     */
/* ------------------------------------------------------------------ */

static inline float fp16_to_f32(uint16_t h)
{
#if defined(__F16C__)
    __m128 v = _mm_cvtph_ps(_mm_cvtsi32_si128((int)h));
    return _mm_cvtss_f32(v);
#else
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t man  = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) { man <<= 1; exp--; }
            man &= 0x3ffu;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    memcpy(&out, &bits, 4);
    return out;
#endif
}

static inline uint16_t f32_to_fp16(float f)
{
#if defined(__F16C__)
    __m128 v = _mm_set_ss(f);
    return (uint16_t)_mm_extract_epi16(_mm_cvtps_ph(v, 0), 0);
#else
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t exp  = (x >> 23) & 0xff;
    uint32_t man  = x & 0x7fffffu;
    uint32_t h;
    if (exp == 0xff) {
        if (man) {
            h = sign | 0x7c00u | 0x200u;
        } else {
            h = sign | 0x7c00u;
        }
    } else {
        int e = (int)exp - 127 + 15;
        if (e >= 31) {
            h = sign | 0x7c00u;
        } else if (e <= 0) {
            if (e < -10) {
                h = sign;
            } else {
                man |= 0x800000u;
                int shift = 14 - e;
                h = sign | (man >> shift);
            }
        } else {
            h = sign | ((uint32_t)e << 10) | (man >> 13);
        }
    }
    return (uint16_t)h;
#endif
}

/* bfloat16 -> fp32 is an exact 16-bit left shift of the stored bit pattern. */
static inline float bf16_to_f32(uint16_t b)
{
    uint32_t bits = (uint32_t)b << 16;
    float out;
    memcpy(&out, &bits, 4);
    return out;
}

/* ------------------------------------------------------------------ */
/* SIMD kernels                                                        */
/* ------------------------------------------------------------------ */

/*
 * C[M][N] = A[M][K] (fp32) * B[N][K] (fp16, row-major)
 * Optionally accumulates: C starts as bias if bias != NULL.
 */
void gemm_fp16_fp32(float *C, const float *A, const uint16_t *B,
                    int M, int K, int N);

/*
 * Same as gemm_fp16_fp32 but B is stored as bfloat16 (native Qwen2 /
 * OLMo weight format); each element is dequantized to fp32 on the fly.
 */
void gemm_bf16_fp32(float *C, const float *A, const uint16_t *B,
                    int M, int K, int N);

/*
 * int8 GEMM with per-row activation scales and per-column weight scales:
 *   C[m][n] = bias[n] + a_scale[m] * b_scale[n] * sum_k A[m][k] * B[n][k]
 * A is int8 [M][K], B is int8 [N][K].
 */
void gemm_int8(float *C, const int8_t *A, const float *a_scale,
               const int8_t *B, const float *b_scale, const float *bias,
               int M, int K, int N);

/* ------------------------------------------------------------------ */
/* Tokenizer (GPT-2 byte-level BPE)                                    */
/* ------------------------------------------------------------------ */

typedef struct {
    char **id_to_token;         /* vocab entries, malloc'd strings */
    int32_t *byte_to_id;        /* [256]: vocab id of each raw byte (byte encoder) */
    int32_t *cp_to_byte;        /* [512]: inverse mapping: codepoint -> raw byte (for decode) */
    uint32_t *merge_l, *merge_r, *merge_m;
    int n_merges;
    int vocab_size;
    int variant;                /* 0 = GPT-2 pretokenizer, 1 = Qwen2 pretokenizer */
    int eos_id;                 /* <|endoftext|> vocab id, or -1 if absent */

    /* Special-token table for encode (longest-match-first). */
    char    **sp_tok;           /* special-token strings (points into id_to_token storage) */
    uint32_t *sp_len;           /* strlen of each special token */
    int32_t  *sp_id;            /* vocab id of each special token */
    int       n_sp;             /* count of special tokens */
} Tokenizer;

int tokenizer_load(Tokenizer *t, const void *vocab_blob, size_t vocab_bytes,
                   const void *merges_blob, size_t merges_bytes, int vocab_size,
                   int variant);
void tokenizer_free(Tokenizer *t);
int tokenizer_encode(const Tokenizer *t, const char *text, int *out, int max_out);
char *tokenizer_decode(const Tokenizer *t, const int *ids, int n);

/* ------------------------------------------------------------------ */
/* Model                                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    char model_name[64];
    int n_layer, n_head, n_embd, n_ctx, vocab;
    int n_kv_head;              /* GQA: KV heads per layer (arch=qwen2) */
    int n_kv_embd;              /* n_kv_head * head_dim */
    int intermediate;           /* SwiGLU MLP hidden size (arch=qwen2) */
    int head_dim;               /* n_embd / n_head */
    int arch;                   /* ARCH_GPT2 | ARCH_QWEN2 */
    size_t n_params;
    double eps;

    /* RoPE (arch=qwen2) */
    double rope_theta;
    int rope_type;              /* ROPE_DEFAULT | ROPE_YARN */
    double yarn_factor, yarn_orig_max, yarn_attention_factor;
    int sliding_window;         /* 0 = full causal attention */
    int tie_word_embeddings;    /* lm_head == wte */
    int eos_token_id;

    const uint16_t *wte;        /* [vocab][n_embd] fp16 */
    const uint16_t *wpe;        /* [n_ctx][n_embd] fp16 (arch=gpt2) or NULL */

    struct LayerW {
        const float *ln1_g, *ln1_b;                 /* n_embd (bias NULL for qwen2) */
        const void *attn_w;                         /* gpt2 int8 or fp16 [3*n_embd][n_embd] */
        const float *attn_s, *attn_b;               /* [3*n_embd] (attn_s only for int8) */
        const void *proj_w;                         /* [n_embd][n_embd] */
        const float *proj_s, *proj_b;
        const float *ln2_g, *ln2_b;
        const void *fc_w;                           /* [4*n_embd][n_embd] */
        const float *fc_s, *fc_b;
        const void *fc2_w;                          /* [n_embd][4*n_embd] */
        const float *fc2_s, *fc2_b;

        /* qwen2 projections: [out][in], no bias */
        const void *q_w, *k_w, *v_w, *o_w;          /* fp16 or int8 */
        const float *q_s, *k_s, *v_s, *o_s;         /* int8 per-output scales */
        const float *q_b, *k_b, *v_b, *o_b;         /* always NULL (bias=False) */
        const void *gate_w, *up_w, *down_w;
        const float *gate_s, *up_s, *down_s;
        const float *gate_b, *up_b, *down_b;
    } *lw;

    const float *lnf_g, *lnf_b;                     /* final norm (bias NULL for qwen2) */
    const uint16_t *lm_head;                        /* [vocab][n_embd] fp16 (untied) or NULL */

    int precision;              /* 0 = int8, 1 = fp16, 2 = bf16 (from .ecm weight dtype) */

    /* KV cache: [n_layer][batch][n_ctx][n_kv_embd] fp16 (n_kv_embd == n_embd for gpt2) */
    uint16_t *k_cache, *v_cache;
    int batch_cap;
    int max_ctx;                /* 0 = use n_ctx from model; otherwise clamp */

    /* workspace */
    float *ws_x, *ws_tmp, *ws_qkv, *ws_attn, *ws_scores;
    int8_t *ws_q;
    float *ws_qs;
    size_t ws_q_cols;           /* int8 activation workspace columns per row */

    Tokenizer tok;

    void *file_data;            /* .ecm file contents (mmap'd, owned) */
    size_t file_size;
    int file_mapped;
} GPT2;

typedef struct {
    char name[128];
    int dtype;                  /* 0=fp32 1=fp16 2=int8 3=text */
    size_t nelems;
    size_t bytes;
    size_t offset;              /* absolute offset in file */
    size_t dims[4];
    int ndims;
} EcmTensor;

int gpt2_load(GPT2 *g, const char *path, int batch_cap, int max_ctx);
void gpt2_free(GPT2 *g);

/*
 * Forward pass for M tokens at the given positions. Logits is [M][vocab].
 * n_past[i] is the current KV length for sequence i; cache rows are indexed
 * by absolute position `positions[i]`.
 *
 * `seqs` (may be NULL) gives the batch slot each token belongs to. When NULL,
 * token m belongs to slot m (batched decode, one token per sequence). When
 * non-NULL, token m is placed in slot seqs[m] at absolute position
 * positions[m] (used for multi-token prefill of a single sequence).
 */
void gpt2_forward(GPT2 *g, const int *tokens, const int *positions,
                  const int *seqs, int M, float *logits);

/* Qwen2 / OLMo-style decoder forward (RoPE + GQA + SwiGLU + RMSNorm). */
void qwen2_forward(GPT2 *g, const int *tokens, const int *positions,
                   const int *seqs, int M, float *logits);

/* Effective context length (respects max_ctx if set) */
static inline int gpt2_eff_ctx(const GPT2 *g) {
    return (g->max_ctx > 0 && g->max_ctx < g->n_ctx) ? g->max_ctx : g->n_ctx;
}

/* Parse the .ecm tensor table. Returns tensor count. */
int ecm_parse(const void *data, size_t size, EcmTensor *out, int max,
              GPT2 *meta_out);

/* ------------------------------------------------------------------ */
/* Utilities                                                           */
/* ------------------------------------------------------------------ */

float peak_rss_gb(void);
double now_sec(void);

#endif
