#include "edgecore.h"

#if defined(__AVX2__)

static inline float hsum_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sum = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sum);
    sum = _mm_add_ss(sum, shuf);
    return _mm_cvtss_f32(sum);
}

static inline int32_t hsum_epi32(__m256i v)
{
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    hi = _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 3, 0, 1));
    lo = _mm_add_epi32(lo, hi);
    hi = _mm_shuffle_epi32(lo, _MM_SHUFFLE(1, 0, 3, 2));
    lo = _mm_add_epi32(lo, hi);
    return _mm_cvtsi128_si32(lo);
}

static inline float dot_fp16_avx2(const float *a, const uint16_t *b, int K)
{
    __m256 acc = _mm256_setzero_ps();
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        __m128i bh = _mm_loadu_si128((const __m128i *)(b + k));
        __m256 bv = _mm256_cvtph_ps(bh);
        __m256 av = _mm256_loadu_ps(a + k);
        acc = _mm256_fmadd_ps(av, bv, acc);
    }
    float res = hsum_ps(acc);
    for (; k < K; k++) res += a[k] * fp16_to_f32(b[k]);
    return res;
}

/*
 * bfloat16 dot product. A bf16 value is the top 16 bits of its fp32
 * representation, so dequantizing to fp32 is a 16-bit left shift on each
 * zero-extended 32-bit lane, then a bit-cast to float.
 */
static inline float dot_bf16_avx2(const float *a, const uint16_t *b, int K)
{
    __m256 acc = _mm256_setzero_ps();
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        __m128i bh = _mm_loadu_si128((const __m128i *)(b + k));
        __m256i b32 = _mm256_cvtepu16_epi32(bh);
        __m256i bits = _mm256_slli_epi32(b32, 16);
        __m256 bv = _mm256_castsi256_ps(bits);
        __m256 av = _mm256_loadu_ps(a + k);
        acc = _mm256_fmadd_ps(av, bv, acc);
    }
    float res = hsum_ps(acc);
    for (; k < K; k++) res += a[k] * bf16_to_f32(b[k]);
    return res;
}

/*
 * Non-saturating int8 dot product (AVX2).
 *
 * The naive `_mm256_maddubs_epi16((a XOR 0x80), b)` trick computes
 * (a+128)*b pairwise into int16, but each product can reach
 * 255*127 = 32385 and a pair can exceed the int16 range (32767),
 * silently saturating and corrupting the result.
 *
 * Instead decompose a = alo - 128*sgn, where alo = a & 0x7f in [0,127]
 * and sgn = 1 when a < 0.  Then:
 *   sum a*b = sum alo*b - 128 * sum_{a<0} b
 * Both `_mm256_maddubs_epi16` terms stay well inside int16
 * (|127*127|*2 = 32258 < 32767), so no saturation occurs.
 */
static inline int32_t dot_int8_avx2(const int8_t *a, const int8_t *b, int K)
{
    __m256i acc  = _mm256_setzero_si256();
    __m256i accm = _mm256_setzero_si256();
    __m256i ones   = _mm256_set1_epi16(1);
    __m256i one8   = _mm256_set1_epi8(1);
    __m256i mask7f = _mm256_set1_epi8(0x7f);
    __m256i zero   = _mm256_setzero_si256();
    int k = 0;
    for (; k + 32 <= K; k += 32) {
        __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + k));
        __m256i b0 = _mm256_loadu_si256((const __m256i *)(b + k));
        __m256i alo   = _mm256_and_si256(a0, mask7f);                 /* 0..127 */
        __m256i amask = _mm256_and_si256(_mm256_cmpgt_epi8(zero, a0), one8);
        __m256i p  = _mm256_maddubs_epi16(alo, b0);   /* sum alo*b per pair   */
        __m256i pm = _mm256_maddubs_epi16(amask, b0); /* sum b where a<0 pair */
        acc  = _mm256_add_epi32(acc,  _mm256_madd_epi16(p,  ones));
        accm = _mm256_add_epi32(accm, _mm256_madd_epi16(pm, ones));
    }
    int32_t res = hsum_epi32(acc) - 128 * hsum_epi32(accm);
    for (; k < K; k++) res += (int32_t)a[k] * (int32_t)b[k];
    return res;
}

#else /* scalar fallback */

static inline float dot_fp16_avx2(const float *a, const uint16_t *b, int K)
{
    float res = 0.0f;
    for (int k = 0; k < K; k++) res += a[k] * fp16_to_f32(b[k]);
    return res;
}

static inline float dot_bf16_avx2(const float *a, const uint16_t *b, int K)
{
    float res = 0.0f;
    for (int k = 0; k < K; k++) res += a[k] * bf16_to_f32(b[k]);
    return res;
}

static inline int32_t dot_int8_avx2(const int8_t *a, const int8_t *b, int K)
{
    int32_t res = 0;
    for (int k = 0; k < K; k++) res += (int32_t)a[k] * (int32_t)b[k];
    return res;
}

#endif

void gemm_fp16_fp32(float *C, const float *A, const uint16_t *B,
                    int M, int K, int N)
{
#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const uint16_t *brow = B + (size_t)n * K;
        for (int m = 0; m < M; m++) {
            C[(size_t)m * N + n] = dot_fp16_avx2(A + (size_t)m * K, brow, K);
        }
    }
}

void gemm_bf16_fp32(float *C, const float *A, const uint16_t *B,
                    int M, int K, int N)
{
#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const uint16_t *brow = B + (size_t)n * K;
        for (int m = 0; m < M; m++) {
            C[(size_t)m * N + n] = dot_bf16_avx2(A + (size_t)m * K, brow, K);
        }
    }
}

void gemm_int8(float *C, const int8_t *A, const float *a_scale,
               const int8_t *B, const float *b_scale, const float *bias,
               int M, int K, int N)
{
#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const int8_t *brow = B + (size_t)n * K;
        float bs = b_scale ? b_scale[n] : 1.0f;
        float bv = bias ? bias[n] : 0.0f;
        for (int m = 0; m < M; m++) {
            int32_t dot = dot_int8_avx2(A + (size_t)m * K, brow, K);
            float asc = a_scale[m];
            C[(size_t)m * N + n] = bv + (float)dot * asc * bs;
        }
    }
}
