#include "edgecore.h"

/* ------------------------------------------------------------------ */
/* Byte classification (regex-lite approximation of GPT-2's pattern)   */
/* ------------------------------------------------------------------ */

static inline int byte_is_space(uint8_t b)
{
    return b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == '\v' || b == '\f';
}

static inline int byte_is_letter(uint8_t b)
{
    return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || b >= 0x80;
}

static inline int byte_is_digit(uint8_t b)
{
    return b >= '0' && b <= '9';
}

/* ------------------------------------------------------------------ */
/* Small growable vectors                                              */
/* ------------------------------------------------------------------ */

typedef struct { int *d; int n, cap; } Ivec;

static void ivec_push(Ivec *v, int x)
{
    if (v->n == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 16;
        v->d = realloc(v->d, (size_t)v->cap * sizeof(int));
        if (!v->d) { fprintf(stderr, "ivec_push: OOM\n"); exit(1); }
    }
    v->d[v->n++] = x;
}

typedef struct { Ivec *pieces; int n, cap; } PieceList;

static void pl_push(PieceList *pl, const uint8_t *s, size_t n)
{
    if (pl->n == pl->cap) {
        pl->cap = pl->cap ? pl->cap * 2 : 16;
        pl->pieces = realloc(pl->pieces, (size_t)pl->cap * sizeof(Ivec));
        if (!pl->pieces) { fprintf(stderr, "pl_push: OOM\n"); exit(1); }
    }
    Ivec *p = &pl->pieces[pl->n++];
    p->d = NULL; p->n = 0; p->cap = 0;
    for (size_t i = 0; i < n; i++) ivec_push(p, s[i]);
}

/* ------------------------------------------------------------------ */
/* Pre-tokenization: split into byte sequences using a regex-lite       */
/* scanner. Approximates:                                              */
/*   's|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ |  */
/*   \s+(?!\S) | \s+                                                   */
/* ------------------------------------------------------------------ */

static PieceList pretokenize(const uint8_t *s, size_t n)
{
    static const char *const ctrs[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    PieceList pl = {NULL, 0, 0};
    size_t i = 0;
    while (i < n) {
        int mlen = 0;
        for (int c = 0; c < 7; c++) {
            size_t cl = strlen(ctrs[c]);
            if (i + cl <= n && memcmp(s + i, ctrs[c], cl) == 0) { mlen = (int)cl; break; }
        }
        if (mlen > 0) {
            pl_push(&pl, s + i, (size_t)mlen);
            i += (size_t)mlen;
            continue;
        }

        int sp = (s[i] == ' ') ? 1 : 0;
        int type = -1; /* 2=letters 3=digits 4=other */
        if (sp) {
            if (i + 1 < n && byte_is_letter(s[i + 1])) type = 2;
            else if (i + 1 < n && byte_is_digit(s[i + 1])) type = 3;
            else if (i + 1 < n && !byte_is_space(s[i + 1])) type = 4;
        } else {
            if (byte_is_letter(s[i])) type = 2;
            else if (byte_is_digit(s[i])) type = 3;
            else if (!byte_is_space(s[i])) type = 4;
        }

        int j = (int)i + sp;
        if (type == 2) {
            while (j < (int)n && byte_is_letter(s[j])) j++;
            pl_push(&pl, s + i, (size_t)(j - (int)i));
            i = (size_t)j;
            continue;
        } else if (type == 3) {
            while (j < (int)n && byte_is_digit(s[j])) j++;
            pl_push(&pl, s + i, (size_t)(j - (int)i));
            i = (size_t)j;
            continue;
        } else if (type == 4) {
            while (j < (int)n && !byte_is_space(s[j]) && !byte_is_letter(s[j]) && !byte_is_digit(s[j])) j++;
            pl_push(&pl, s + i, (size_t)(j - (int)i));
            i = (size_t)j;
            continue;
        }

        /* whitespace run */
        j = (int)i;
        while (j < (int)n && byte_is_space(s[j])) j++;
        pl_push(&pl, s + i, (size_t)(j - (int)i));
        i = (size_t)j;
    }
    return pl;
}

static void pl_free(PieceList *pl)
{
    for (int i = 0; i < pl->n; i++) free(pl->pieces[i].d);
    free(pl->pieces);
}

/* ------------------------------------------------------------------ */
/* String hash map (token content -> id)                               */
/* ------------------------------------------------------------------ */

typedef struct { const char *key; uint32_t len; int32_t id; } SMEntry;
typedef struct { SMEntry *e; size_t cap, count; } StrMap;

static uint64_t fnv1a(const void *data, size_t len)
{
    const uint8_t *p = data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static void sm_init(StrMap *m, size_t cap)
{
    size_t c = 16;
    while (c < cap) c *= 2;
    m->e = calloc(c, sizeof(SMEntry));
    m->cap = c;
    m->count = 0;
    if (!m->e) { fprintf(stderr, "StrMap: OOM\n"); exit(1); }
    for (size_t i = 0; i < c; i++) m->e[i].id = -1;
}

static void sm_put(StrMap *m, const char *key, uint32_t len, int32_t id)
{
    if (m->count * 2 >= m->cap) {
        StrMap old = *m;
        sm_init(m, old.cap * 2);
        for (size_t i = 0; i < old.cap; i++)
            if (old.e[i].id >= 0) sm_put(m, old.e[i].key, old.e[i].len, old.e[i].id);
        free(old.e);
    }
    size_t idx = (size_t)(fnv1a(key, len) & (m->cap - 1));
    while (m->e[idx].id >= 0) idx = (idx + 1) & (m->cap - 1);
    m->e[idx].key = key;
    m->e[idx].len = len;
    m->e[idx].id = id;
    m->count++;
}

static int32_t sm_get(const StrMap *m, const char *key, uint32_t len)
{
    if (!m->cap) return -1;
    size_t idx = (size_t)(fnv1a(key, len) & (m->cap - 1));
    size_t start = idx;
    while (m->e[idx].id >= 0) {
        if (m->e[idx].len == len && memcmp(m->e[idx].key, key, len) == 0) return m->e[idx].id;
        idx = (idx + 1) & (m->cap - 1);
        if (idx == start) break;
    }
    return -1;
}

/* GPT-2 bytes_to_unicode: map each raw byte to the unicode codepoint used as
 * its vocab key, then resolve that to the vocab id via the StrMap. */
static void build_byte_encoder(StrMap *map, int32_t *byte_to_id)
{
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int in_bs = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        uint32_t cp = (uint32_t)(in_bs ? b : 256 + n);
        if (!in_bs) n++;
        uint8_t u8[4];
        int ulen;
        if (cp < 0x80) { u8[0] = (uint8_t)cp; ulen = 1; }
        else if (cp < 0x800) {
            u8[0] = (uint8_t)(0xC0 | (cp >> 6));
            u8[1] = (uint8_t)(0x80 | (cp & 0x3F));
            ulen = 2;
        } else {
            u8[0] = (uint8_t)(0xE0 | (cp >> 12));
            u8[1] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
            u8[2] = (uint8_t)(0x80 | (cp & 0x3F));
            ulen = 3;
        }
        byte_to_id[b] = sm_get(map, (const char *)u8, (uint32_t)ulen);
    }
}

/* ------------------------------------------------------------------ */
/* Blob readers                                                         */
/* ------------------------------------------------------------------ */

static inline uint32_t rd_u32(const uint8_t **pp)
{
    uint32_t v;
    memcpy(&v, *pp, 4);
    *pp += 4;
    return v;
}

/* ------------------------------------------------------------------ */
/* Load                                                                */
/* ------------------------------------------------------------------ */

int tokenizer_load(Tokenizer *t, const void *vocab_blob, size_t vocab_bytes,
                   const void *merges_blob, size_t merges_bytes, int vocab_size)
{
    memset(t, 0, sizeof(*t));
    t->vocab_size = vocab_size;
    const uint8_t *p = vocab_blob;
    const uint8_t *end = p + vocab_bytes;

    uint32_t count = rd_u32(&p);
    t->id_to_token = calloc((size_t)vocab_size, sizeof(char *));
    if (!t->id_to_token) return -1;

    StrMap map = {NULL, 0, 0};
    sm_init(&map, (size_t)count * 2);

    for (uint32_t i = 0; i < count && p + 8 <= end; i++) {
        uint32_t id = rd_u32(&p);
        uint32_t len = rd_u32(&p);
        if (p + len > end) break;
        char *s = malloc((size_t)len + 1);
        memcpy(s, p, len);
        s[len] = '\0';
        p += len;
        if (id < (uint32_t)vocab_size) {
            t->id_to_token[id] = s;
            sm_put(&map, s, len, (int32_t)id);
        } else {
            free(s);
        }
    }

    /* merges: each entry is (lenA, A, lenB, B); store (idA, idB, idM). */
    p = merges_blob;
    end = p + merges_bytes;
    uint32_t nmerges = rd_u32(&p);
    t->n_merges = (int)nmerges;
    t->merge_l = malloc((size_t)nmerges * sizeof(uint32_t));
    t->merge_r = malloc((size_t)nmerges * sizeof(uint32_t));
    t->merge_m = malloc((size_t)nmerges * sizeof(uint32_t));
    if (!t->merge_l || !t->merge_r || !t->merge_m) return -1;

    int used = 0;
    char *cat = NULL;
    size_t cat_cap = 0;
    for (uint32_t i = 0; i < nmerges && p + 8 <= end; i++) {
        uint32_t la = rd_u32(&p);
        if (p + la > end) break;
        const char *a = (const char *)p;
        p += la;
        uint32_t lb = rd_u32(&p);
        if (p + lb > end) break;
        const char *b = (const char *)p;
        p += lb;

        int32_t ida = sm_get(&map, a, la);
        int32_t idb = sm_get(&map, b, lb);
        size_t need = (size_t)la + lb;
        if (need + 1 > cat_cap) {
            cat_cap = need + 1;
            cat = realloc(cat, cat_cap);
        }
        memcpy(cat, a, la);
        memcpy(cat + la, b, lb);
        int32_t idm = sm_get(&map, cat, need);
        if (ida >= 0 && idb >= 0 && idm >= 0) {
            t->merge_l[used] = (uint32_t)ida;
            t->merge_r[used] = (uint32_t)idb;
            t->merge_m[used] = (uint32_t)idm;
            used++;
        }
    }
    t->n_merges = used;
    free(cat);

    t->byte_to_id = malloc(256 * sizeof(int32_t));
    if (!t->byte_to_id) return -1;
    build_byte_encoder(&map, t->byte_to_id);
    free(map.e);
    return 0;
}

void tokenizer_free(Tokenizer *t)
{
    if (t->id_to_token) {
        for (int i = 0; i < t->vocab_size; i++) free(t->id_to_token[i]);
        free(t->id_to_token);
    }
    free(t->byte_to_id);
    free(t->merge_l);
    free(t->merge_r);
    free(t->merge_m);
    memset(t, 0, sizeof(*t));
}

/* ------------------------------------------------------------------ */
/* Encode / decode                                                      */
/* ------------------------------------------------------------------ */

int tokenizer_encode(const Tokenizer *t, const char *text, int *out, int max_out)
{
    size_t n = strlen(text);
    if (strcmp(text, "<|endoftext|>") == 0) {
        if (max_out < 1) return -1;
        out[0] = t->vocab_size - 1;
        return 1;
    }

    PieceList pl = pretokenize((const uint8_t *)text, n);
    int total = 0;
    int err = 0;
    for (int p = 0; p < pl.n && !err; p++) {
        Ivec ids = {0};
        for (int i = 0; i < pl.pieces[p].n; i++) {
            int b = pl.pieces[p].d[i];
            int bid = t->byte_to_id ? t->byte_to_id[b] : -1;
            ivec_push(&ids, bid >= 0 ? bid : b);
        }
        for (int r = 0; r < t->n_merges && ids.n > 1; r++) {
            uint32_t l = t->merge_l[r], rr = t->merge_r[r], m = t->merge_m[r];
            Ivec next = {0};
            for (int i = 0; i < ids.n;) {
                if (i + 1 < ids.n && (uint32_t)ids.d[i] == l && (uint32_t)ids.d[i + 1] == rr) {
                    ivec_push(&next, (int)m);
                    i += 2;
                } else {
                    ivec_push(&next, ids.d[i]);
                    i++;
                }
            }
            free(ids.d);
            ids = next;
        }
        if (total + ids.n > max_out) {
            err = 1;
        } else {
            memcpy(out + total, ids.d, (size_t)ids.n * sizeof(int));
            total += ids.n;
        }
        free(ids.d);
    }
    pl_free(&pl);
    return err ? -1 : total;
}

char *tokenizer_decode(const Tokenizer *t, const int *ids, int n)
{
    size_t total = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id < 0 || id >= t->vocab_size || !t->id_to_token[id]) id = 0;
        total += strlen(t->id_to_token[id]);
    }
    char *out = malloc(total + 1);
    size_t o = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id < 0 || id >= t->vocab_size || !t->id_to_token[id]) id = 0;
        size_t l = strlen(t->id_to_token[id]);
        memcpy(out + o, t->id_to_token[id], l);
        o += l;
    }
    out[o] = '\0';
    return out;
}
