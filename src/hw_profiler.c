#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <cpuid.h>

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static const char *read_first(const char *path, char *buf, size_t n)
{
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    if (!fgets(buf, (int)n, f)) { fclose(f); return NULL; }
    fclose(f);
    char *nl = strchr(buf, '\n');
    if (nl) *nl = '\0';
    return buf;
}

static int cpu_simd_flags(char *out, size_t cap)
{
    unsigned eax, ebx, ecx, edx;
    size_t o = 0;
    int has = 0;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        if (edx & (1u << 25)) { has = 1; o += (size_t)snprintf(out + o, cap - o, "%s", o ? " sse" : "sse"); }
        if (edx & (1u << 26)) { has = 1; o += (size_t)snprintf(out + o, cap - o, "%s", o ? " sse2" : "sse2"); }
        if (ecx & (1u << 19)) { has = 1; o += (size_t)snprintf(out + o, cap - o, "%s", o ? " sse4.1" : "sse4.1"); }
        if (ecx & (1u << 20)) { has = 1; o += (size_t)snprintf(out + o, cap - o, "%s", o ? " sse4.2" : "sse4.2"); }
        if (ecx & (1u << 12)) { has = 1; o += (size_t)snprintf(out + o, cap - o, "%s", o ? " fma" : "fma"); }
        if (ecx & (1u << 29)) { has = 1; o += (size_t)snprintf(out + o, cap - o, "%s", o ? " f16c" : "f16c"); }
        if (ecx & (1u << 27)) { has = 1; o += (size_t)snprintf(out + o, cap - o, "%s", o ? " avx" : "avx"); }
        unsigned ex, eb, ec, ed;
        if (__get_cpuid_count(7, 0, &ex, &eb, &ec, &ed)) {
            if (eb & (1u << 5)) { has = 1; o += (size_t)snprintf(out + o, cap - o, "%s", o ? " avx2" : "avx2"); }
            if (eb & (1u << 16)) { has = 1; o += (size_t)snprintf(out + o, cap - o, "%s", o ? " avx512f" : "avx512f"); }
            if (eb & (1u << 30)) { has = 1; o += (size_t)snprintf(out + o, cap - o, "%s", o ? " avx512bw" : "avx512bw"); }
        }
    }
    return has;
}

/* physical core count: unique (physical id, core id) pairs in /proc/cpuinfo */
static long physical_cores(void)
{
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return -1;
    char line[256];
    typedef struct { long phys, core; } PC;
    PC seen[512];
    int n = 0;
    long cur_phys = -1, cur_core = -1;
    while (fgets(line, sizeof line, f)) {
        long v;
        if (sscanf(line, "physical id : %ld", &v) == 1) cur_phys = v;
        else if (sscanf(line, "core id : %ld", &v) == 1) cur_core = v;
        if (cur_phys >= 0 && cur_core >= 0) {
            int found = 0;
            for (int i = 0; i < n; i++)
                if (seen[i].phys == cur_phys && seen[i].core == cur_core) { found = 1; break; }
            if (!found && n < 512) { seen[n].phys = cur_phys; seen[n].core = cur_core; n++; }
        }
    }
    fclose(f);
    return n > 0 ? n : -1;
}

/* NUMA node count from /sys/devices/system/node */
static int numa_nodes(void)
{
    FILE *f = fopen("/sys/devices/system/node/online", "r");
    if (!f) return 1;
    char buf[128] = "";
    if (!fgets(buf, sizeof buf, f)) { fclose(f); return 1; }
    fclose(f);
    int n = 0;
    const char *p = buf;
    while (*p) {
        if (*p == '-') {
            /* range a-b */
            long lo = 0, hi = 0;
            sscanf(p + 1, "%ld-%ld", &lo, &hi);
            n += (int)(hi - lo) + 1;
            while (*p && *p != ',') p++;
        } else if (*p != ',') {
            if (*p != '\n') n++;
            p++;
        } else {
            p++;
        }
    }
    return n > 0 ? n : 1;
}

/* crude memory bandwidth estimate: copy 64MB with rep movsb style loop */
static double mem_bw_gbps(void)
{
    size_t n = 64u << 20;
    char *a = malloc(n), *b = malloc(n);
    if (!a || !b) { free(a); free(b); return 0; }
    memset(a, 1, n);
    memset(b, 0, n);
    double t0 = now_sec();
    for (int r = 0; r < 8; r++) {
        for (size_t i = 0; i < n; i += 4096) {
            memcpy(b + i, a + i, 4096);
        }
    }
    double dt = now_sec() - t0;
    double bytes = (double)n * 8.0 * 2.0; /* read + write */
    free(a);
    free(b);
    return dt > 0 ? bytes / dt / 1e9 : 0;
}

/* enumerate cache hierarchy from sysfs: index != level, so read each index */
typedef struct { int level; char size[64]; char type[64]; } CacheEnt;

static void emit_caches(void)
{
    CacheEnt c[16];
    int n = 0;
    for (int idx = 0; idx < 16 && n < 16; idx++) {
        char path[160];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/level", idx);
        char lvl[32];
        if (!read_first(path, lvl, sizeof lvl)) break;
        int level = atoi(lvl);
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/size", idx);
        if (!read_first(path, c[n].size, sizeof c[n].size)) continue;
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/type", idx);
        if (!read_first(path, c[n].type, sizeof c[n].type)) snprintf(c[n].type, sizeof c[n].type, "Unified");
        c[n].level = level;
        n++;
    }
    /* stable sort by level then type */
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            int swap = 0;
            if (c[j].level < c[i].level) swap = 1;
            else if (c[j].level == c[i].level && strcmp(c[j].type, c[i].type) < 0) swap = 1;
            if (swap) { CacheEnt t = c[i]; c[i] = c[j]; c[j] = t; }
        }
    printf("  \"caches\": [\n");
    for (int i = 0; i < n; i++)
        printf("%s    {\"level\": %d, \"size\": \"%s\", \"type\": \"%s\"}",
               i ? ",\n" : "", c[i].level, c[i].size, c[i].type);
    printf("\n  ],\n");
}

static void emit_json(void)
{
    char buf[512];
    char name[128] = "unknown";
    char freq[64] = "0";
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        while (fgets(buf, sizeof buf, f)) {
            if (!strncmp(buf, "model name", 10)) {
                char *p = strchr(buf, ':');
                if (p) { p++; while (*p == ' ') p++; char *nl = strchr(p, '\n'); if (nl) *nl = '\0'; snprintf(name, sizeof name, "%s", p); break; }
            }
        }
        fclose(f);
    }
    /* frequency from the last cpuinfo cpu MHz line */
    {
        FILE *ff = fopen("/proc/cpuinfo", "r");
        if (ff) {
            while (fgets(buf, sizeof buf, ff)) {
                if (!strncmp(buf, "cpu MHz", 7)) {
                    char *p = strchr(buf, ':');
                    if (p) {
                        p++;
                        while (*p == ' ') p++;
                        char *nl = strchr(p, '\n');
                        if (nl) *nl = '\0';
                        snprintf(freq, sizeof freq, "%s", p);
                    }
                }
            }
            fclose(ff);
        }
    }

    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    long nphys = physical_cores();
    if (nphys <= 0) nphys = nproc;
    long pages = sysconf(_SC_PHYS_PAGES);
    long psize = sysconf(_SC_PAGE_SIZE);
    double ram_gb = (double)pages * psize / 1e9;
    int nnodes = numa_nodes();

    struct sysinfo si;
    sysinfo(&si);

    char simd[256] = "";
    cpu_simd_flags(simd, sizeof simd);

    /* cache hierarchy from sysfs */
    printf("{\n");
    printf("  \"platform\": \"linux\",\n");
    printf("  \"cpu_model\": "); 
    printf("\""); 
    for (char *p = name; *p; p++) { if (*p == '"') printf("\\\""); else putchar(*p); }
    printf("\",\n");
    printf("  \"cpu_freq_mhz\": %s,\n", freq[0] ? freq : "0");
    printf("  \"physical_cores\": %ld,\n", nphys);
    printf("  \"logical_cores\": %ld,\n", nproc);
    printf("  \"threads_per_core\": %ld,\n", nphys > 0 ? (nproc + nphys - 1) / nphys : 1);
    printf("  \"numa_nodes\": %d,\n", nnodes);
    printf("  \"ram_gb\": %.2f,\n", ram_gb);
    printf("  \"simd\": \"%s\",\n", simd);
    emit_caches();
    printf("  \"memory_bandwidth_gbps\": %.2f\n", mem_bw_gbps());
    printf("}\n");
}

int main(void)
{
    emit_json();
    return 0;
}
