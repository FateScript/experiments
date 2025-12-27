#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

// gcc loop_unroll.c; ./a.out

// result on my machine:
// loop1, BEST CPE = 6.565, AVG CPE = 6.595
// loop2, BEST CPE = 3.512, AVG CPE = 3.537
// loop4, BEST CPE = 2.928, AVG CPE = 2.978


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


void loop1(float *a, long n) {
    for (long i = 0; i < n; i++) {
        a[i] = a[i] + 1;
    }
}

void loop2(float* a, long n) {
    for (long i = 1; i < n-1; i+=2) {
        a[i] = a[i] + 1;
        a[i+1] = a[i+1] + 1;
    }
}

void loop4(float* a, long n) {
    for (long i = 1; i < n-1; i+=4) {
        a[i] = a[i] + 1;
        a[i+1] = a[i+1] + 1;
        a[i+2] = a[i+2] + 1;
        a[i+3] = a[i+3] + 1;
    }
}

int main() {
    long n = 1 << 20; // larger n
    float *a = aligned_alloc(64, n * sizeof(float));

    for (long i = 0; i < n; ++i) a[i] = 1.0f;

    // warm-up
    loop1(a, n);
    loop2(a, n);
    loop4(a, n);

    const int runs = 10;
    uint64_t best1 = UINT64_MAX, best2 = UINT64_MAX, best3 = UINT64_MAX;
    uint64_t sum1 = 0, sum2 = 0, sum3 = 0;
    for (int r = 0; r < runs; ++r) {
        uint64_t start = rdtsc_serialized();
        loop1(a, n);
        uint64_t end = rdtsc_serialized();
        uint64_t cycles1 = end - start;
        sum1 += cycles1;
        if (cycles1 < best1) best1 = cycles1;

        start = rdtsc_serialized();
        loop2(a, n);
        end = rdtsc_serialized();
        uint64_t cycles2 = end - start;
        sum2 += cycles2;
        if (cycles2 < best2) best2 = cycles2;

        start = rdtsc_serialized();
        loop4(a, n);
        end = rdtsc_serialized();
        uint64_t cycles3 = end - start;
        sum3 += cycles3;
        if (cycles3 < best3) best3 = cycles3;
    }

    double avg_cpe1 = (double)sum1 / runs / n;
    double avg_cpe2 = (double)sum2 / runs / n;
    double avg_cpe3 = (double)sum3 / runs / n;

    double cpe1 = (double)best1 / n;
    double cpe2 = (double)best2 / n;
    double cpe3 = (double)best3 / n;

    printf("loop1, BEST CPE = %.3f, AVG CPE = %.3f\n", cpe1, avg_cpe1);
    printf("loop2, BEST CPE = %.3f, AVG CPE = %.3f\n", cpe2, avg_cpe2);
    printf("loop4, BEST CPE = %.3f, AVG CPE = %.3f\n", cpe3, avg_cpe3);

    free(a);
    return 0;
}
