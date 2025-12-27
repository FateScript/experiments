#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

// gcc cpe.c; ./a.out
// prefix sum1, BEST CPE = 7.902, AVG CPE = 7.931
// prefix sum2, BEST CPE = 7.500, AVG CPE = 7.523

// gcc -O3 cpe.c; ./a.out
// prefix sum1, BEST CPE = 3.116, AVG CPE = 3.152
// prefix sum2, BEST CPE = 4.215, AVG CPE = 4.266


static inline uint64_t rdtsc() {
    unsigned hi, lo;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t rdtsc_serialized() {
    unsigned hi, lo;
    __asm__ volatile (
        "cpuid\n\t"
        "rdtsc\n\t"
        : "=a"(lo), "=d"(hi)
        :
        : "%rbx", "%rcx"
    );
    return ((uint64_t)hi << 32) | lo;
}


void prefix_sum1(float *a, float *b, long n) {
    b[0] = a[0];
    for (long i = 1; i < n; i++) {
        b[i] = b[i-1] + a[i];
    }
}

void prefix_sum2(float* a, float *b, long n) {
    b[0] = a[0];
    long i;
    for (i = 1; i < n-1; i+=2) {
        float mid_val = b[i-1] + a[i];
        b[i] = mid_val;
        b[i+1] = mid_val + a[i+1];
    }
    if (i < n) b[i] = b[i-1] + a[i];
}

int main() {
    long n = 1 << 20; // larger n
    float *a = aligned_alloc(64, n * sizeof(float));
    float *b = aligned_alloc(64, n * sizeof(float));

    for (long i = 0; i < n; ++i) a[i] = 1.0f;

    // warm-up
    prefix_sum1(a, b, n);
    prefix_sum2(a, b, n);

    const int runs = 10;
    uint64_t best1 = UINT64_MAX, best2 = UINT64_MAX;
    uint64_t sum1 = 0, sum2 = 0;
    for (int r = 0; r < runs; ++r) {
        uint64_t start = rdtsc_serialized();
        prefix_sum1(a, b, n);
        uint64_t end = rdtsc_serialized();
        uint64_t cycles1 = end - start;
        sum1 += cycles1;
        if (cycles1 < best1) best1 = cycles1;

        start = rdtsc_serialized();
        prefix_sum2(a, b, n);
        end = rdtsc_serialized();
        uint64_t cycles2 = end - start;
        sum2 += cycles2;
        if (cycles2 < best2) best2 = cycles2;
    }

    double avg_cpe1 = (double)sum1 / runs / n;
    double avg_cpe2 = (double)sum2 / runs / n;
    double cpe1 = (double)best1 / n;
    double cpe2 = (double)best2 / n;
    printf("prefix sum1, BEST CPE = %.3f, AVG CPE = %.3f\n", cpe1, avg_cpe1);
    printf("prefix sum2, BEST CPE = %.3f, AVG CPE = %.3f\n", cpe2, avg_cpe2);

    free(a); free(b);
    return 0;
}
