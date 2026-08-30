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
    uint32_t *merge_l, *merge_r, *merge_m;
    int n_merges;
    int vocab_size;
} Tokenizer;

int tokenizer_load(Tokenizer *t, const void *vocab_blob, size_t vocab_bytes,
                   const void *merges_blob, size_t merges_bytes, int vocab_size);
void tokenizer_free(Tokenizer *t);
int tokenizer_encode(const Tokenizer *t, const char *text, int *out, int max_out);
char *tokenizer_decode(const Tokenizer *t, const int *ids, int n);

/* ------------------------------------------------------------------ */
/* Model                                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    char model_name[64];
    int n_layer, n_head, n_embd, n_ctx, vocab;
    size_t n_params;
    double eps;

    const uint16_t *wte;        /* [vocab][n_embd] fp16 */
    const uint16_t *wpe;        /* [n_ctx][n_embd] fp16 */

    struct LayerW {
        const float *ln1_g, *ln1_b;                 /* n_embd */
        const void *attn_w;                         /* int8 or fp16 [3*n_embd][n_embd] */
        const float *attn_s, *attn_b;               /* [3*n_embd] (attn_s only for int8) */
        const void *proj_w;                         /* [n_embd][n_embd] */
        const float *proj_s, *proj_b;
        const float *ln2_g, *ln2_b;
        const void *fc_w;                           /* [4*n_embd][n_embd] */
        const float *fc_s, *fc_b;
        const void *fc2_w;                          /* [n_embd][4*n_embd] */
        const float *fc2_s, *fc2_b;
    } *lw;

    const float *lnf_g, *lnf_b;

    int precision;              /* 0 = int8, 1 = fp16 (from .ecm weight dtype) */

    /* KV cache: [n_layer][batch][n_ctx][n_embd] fp16 */
    uint16_t *k_cache, *v_cache;
    int batch_cap;

    /* workspace */
    float *ws_x, *ws_tmp, *ws_qkv, *ws_attn, *ws_scores;
    int8_t *ws_q;
    float *ws_qs;

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

int gpt2_load(GPT2 *g, const char *path, int batch_cap);
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

/* Parse the .ecm tensor table. Returns tensor count. */
int ecm_parse(const void *data, size_t size, EcmTensor *out, int max,
              GPT2 *meta_out);

/* ------------------------------------------------------------------ */
/* Utilities                                                           */
/* ------------------------------------------------------------------ */

float peak_rss_gb(void);
double now_sec(void);

#endif
