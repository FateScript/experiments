#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

// A algorithm that takes fewer floating point operations might be slower.
// Benchmark a straightforward method (poly) vs Horner's method (poly_horner)

// gcc cpe.c; ./a.out
// Warm-up: poly = 1.428571, poly_horner = 1.428571
// poly, BEST CPE = 7.814, AVG CPE = 8.103
// poly horner, BEST CPE = 10.038, AVG CPE = 10.380

// gcc -O3 cpe.c; ./a.out
// Warm-up: poly = 1.428571, poly_horner = 1.428571
// poly, BEST CPE = 0.035, AVG CPE = 0.037
// poly horner, BEST CPE = 0.035, AVG CPE = 0.036

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


float poly(float *a, float x, long n) {
    float result = a[0];
    float x_pow = x;
    for (long i = 1; i <= n; i++) {
        result += a[i] * x_pow;
        x_pow *= x;
    }
    return result;
}

float poly_horner(float* a, float x, long n) {
    float result = a[n];
    for (long i = n-1; i >= 0; i--) {
        result = a[i] + x * result;
    }
    return result;
}

int main() {
    long n = 1 << 15;
    float x = 0.3f;
    float *a = aligned_alloc(64, n * sizeof(float));

    for (long i = 0; i < n; ++i) a[i] = 1.0f;

    // warm-up
    float val1 = poly(a, x, n);
    float val2 = poly_horner(a, x, n);
    printf("Warm-up: poly = %f, poly_horner = %f\n", val1, val2);

    const int runs = 10;
    uint64_t best1 = UINT64_MAX, best2 = UINT64_MAX;
    uint64_t sum1 = 0, sum2 = 0;
    for (int r = 0; r < runs; ++r) {
        uint64_t start = rdtsc_serialized();
        poly(a, x, n);
        uint64_t end = rdtsc_serialized();
        uint64_t cycles1 = end - start;
        sum1 += cycles1;
        if (cycles1 < best1) best1 = cycles1;

        start = rdtsc_serialized();
        poly_horner(a, x, n);
        end = rdtsc_serialized();
        uint64_t cycles2 = end - start;
        sum2 += cycles2;
        if (cycles2 < best2) best2 = cycles2;
    }

    double avg_cpe1 = (double)sum1 / runs / n;
    double avg_cpe2 = (double)sum2 / runs / n;
    double cpe1 = (double)best1 / n;
    double cpe2 = (double)best2 / n;
    printf("poly, BEST CPE = %.3f, AVG CPE = %.3f\n", cpe1, avg_cpe1);
    printf("poly horner, BEST CPE = %.3f, AVG CPE = %.3f\n", cpe2, avg_cpe2);

    free(a);
    return 0;
}
