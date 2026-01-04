#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>

// g++ roofline.cpp -o roofline -O3 -march=native

using namespace std;
using namespace chrono;

struct TestResult {
    int flops_per_element;
    double arithmetic_intensity;
    double time_ms;
    double gflops;
};

class MemoryComputeBoundTest {
private:
    const size_t size = 32 * 1024 * 1024; // 32M data
    const int iterations = 8;
    double* input_a = nullptr;
    double* input_b = nullptr;
    double* output  = nullptr;

    double memory_bandwidth = 0.0;
    std::vector<TestResult> results;
    
    void initialize_data() {
        input_a = new double[size];
        input_b = new double[size];
        output  = new double[size];

        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(0.1, 1.0);
        
        for (size_t i = 0; i < size; i++) {
            input_a[i] = dis(gen);
            input_b[i] = dis(gen);
        }
    }
    
    template<typename Func>
    double time_function(Func f, int warmup = 3) {
        for (int i = 0; i < warmup; i++) {
            f();
        }

        auto start = high_resolution_clock::now();
        f();
        auto end = high_resolution_clock::now();
        
        return duration<double, milli>(end - start).count();
    }
    
public:
    MemoryComputeBoundTest() {
        initialize_data();
    }
    
    void test_memory_copy() {
        auto compute = [&]() {
            for (int iter = 0; iter < iterations; iter++) {
                for (size_t i = 0; i < size; i++) {
                    output[i] = input_a[i];
                }
            }
        };

        double time_ms = time_function(compute);
        double bytes = size * 8.0 * 2.0 * iterations; // read + write
        double bandwidth = bytes / (time_ms / 1000.0) / 1e9;

        memory_bandwidth = bandwidth;

        results.push_back({0, 0.0, time_ms, 0.0});
        
        cout << "Test - Memory Copy (0 FLOPS): " << time_ms << " ms, Bandwidth: " << bandwidth << " GB/s" << endl;
    }
    
    void test_1_flop() {
        auto compute = [&]() {
            for (int iter = 0; iter < iterations; iter++) {
                for (size_t i = 0; i < size; i++) {
                    output[i] = input_a[i] * input_b[i];
                }
            }
        };
        
        double time_ms = time_function(compute);
        double flops = size * iterations * 1.0;
        double gflops = flops / (time_ms / 1000.0) / 1e9;
        
        // AI = 1 FLOP / (2 reads + 1 write) * 8 bytes = 1/24
        double ai = 1.0 / 24.0;
        
        results.push_back({
            1,
            ai,
            time_ms,
            gflops,
        });
        
        cout << "Test - Add (1 FLOP): " << time_ms << " ms, " << gflops << " GFLOPS" << endl;
    }
    
    void test_2_flops() {
        auto compute = [&]() {
            for (int iter = 0; iter < iterations; iter++) {
                for (size_t i = 0; i < size; i++) {
                    output[i] = input_a[i] * input_b[i] + input_a[i];
                }
            }
        };
        
        double time_ms = time_function(compute);
        double flops = size * iterations * 2.0;
        double gflops = flops / (time_ms / 1000.0) / 1e9;
        double ai = 2.0 / 24.0;
        
        results.push_back({2, ai, time_ms, gflops});
        
        cout << "Test - FMA (2 FLOPs): " << time_ms << " ms, " << gflops << " GFLOPS" << endl;
    }

    void test_4_flops() {
        auto compute = [&]() {
            for (int iter = 0; iter < iterations; iter++) {
                for (size_t i = 0; i < size; i++) {
                    double temp = input_a[i] * input_b[i];
                    output[i] = temp * temp + input_a[i] + input_b[i];
                }
            }
        };
        
        double time_ms = time_function(compute);
        double flops = size * iterations * 4.0;
        double gflops = flops / (time_ms / 1000.0) / 1e9;
        double ai = 4.0 / 24.0;
        
        results.push_back({4, ai, time_ms, gflops});

        cout << "Test - 4 FLOPs: " << time_ms << " ms, " << gflops << " GFLOPS" << endl;
    }

    // Generic tester: generate exactly n FLOPs per element
    void test_flops(int n) {
        if (n % 2 == 1) {
            cout << "Invalid FLOPs count: " << n << endl;
            return;
        }

        auto compute = [&, n]() {
            int pairs = n / 2;
            for (int iter = 0; iter < iterations; iter++) {
                for (size_t i = 0; i < size; i++) {
                    double a = input_a[i];
                    double b = input_b[i];

                    // Start with one multiply if n >= 1
                    double result = a * b; // 1 flop

                    // Each loop below does 2 FLOPs: multiply + add
                    for (int j = 1; j < pairs; j++) {
                        result = result * a + b; // 2 flops
                    }

                    // one remaining flop, do an extra multiply
                    result = result + a; // 1 flop
                    output[i] = result;
                }
            }
        };

        double time_ms = time_function(compute);
        double flops = static_cast<double>(size) * iterations * static_cast<double>(n);
        double gflops = flops / (time_ms / 1000.0) / 1e9;
        double ai = static_cast<double>(n) / 24.0; // bytes per element = 2 reads + 1 write = 24

        results.push_back({n, ai, time_ms, gflops});

        cout << "Test - " << n << " FLOPs: " << time_ms << " ms, " << gflops << " GFLOPS" << endl;
    }

    void run_all_tests() {
        cout << "==========================================================" << endl;
        cout << "  Roofline modeling Verification " << endl;
        cout << "==========================================================" << endl;
        cout << "Array size: " << size << " elements (" 
             << (size * 8.0 / 1024 / 1024) << " MB per array)" << endl;
        cout << "Iterations: " << iterations << endl;
        cout << "Total memory access: " << (size * 8.0 * 3.0 * iterations / 1024 / 1024 / 1024) 
             << " GB (read 2 arrays + write 1 array)" << endl;
        cout << "==========================================================" << endl << endl;
        
        test_memory_copy();
        test_1_flop();
        test_2_flops();
        test_4_flops();
        int flop_counts[] = {8, 16, 24, 32, 40, 48, 56, 64, 96, 128};
        for (int n : flop_counts) {
            test_flops(n);
        }
    }
    
    void analyze_results() {
        cout << "==========================================================" << endl;
        cout << "  Analysis: Memory Bound vs Compute Bound Transition" << endl;
        cout << "==========================================================" << endl << endl;
        
        double baseline_time = results[0].time_ms;
        
        cout << left << setw(15) << "FLOPs/elem" 
             << setw(15) << "AI" 
             << setw(15) << "Time(ms)"
             << setw(15) << "Time Ratio"
             << setw(15) << "GFLOPS" << endl;
        cout << string(75, '-') << endl;
        
        for (size_t i = 0; i < results.size(); i++) {
            const auto& r = results[i];
            double time_ratio = r.time_ms / baseline_time;
            
            cout << left << setw(15) << r.flops_per_element
                 << setw(15) << fixed << setprecision(3) << r.arithmetic_intensity
                 << setw(15) << fixed << setprecision(2) << r.time_ms
                 << setw(15) << fixed << setprecision(2) << time_ratio
                 << setw(15) << fixed << setprecision(2) << r.gflops << endl;
        }

        cout << endl;
        
        double estimated_peak = 0.0;
        for (const auto& r : results) {
            if (r.gflops > estimated_peak) {
                estimated_peak = r.gflops;
            }
        }
        cout << "Estimated System Parameters:" << endl;
        cout << "  Memory Bandwidth: " << memory_bandwidth << " GB/s" << endl;
        cout << "  Peak Performance: ~" << estimated_peak << " GFLOPS" << endl;
        cout << endl;

        cout << "==========================================================" << endl;
    }
    
    void save_csv() {
        ofstream csv("transition_results.csv");
        csv << "FLOPs_per_element,Arithmetic_Intensity,Time_ms,GFLOPS\n";

        for (const auto& r : results) {
            csv << r.flops_per_element << ","
                << r.arithmetic_intensity << ","
                << r.time_ms << ","
                << r.gflops << "," << "\n";
        }

        csv.close();
        cout << "Results saved to: transition_results.csv" << endl;
    }
};

int main() {
    MemoryComputeBoundTest test;

    test.run_all_tests();
    test.analyze_results();
    test.save_csv();

    cout << "\nExperiment complete!" << endl;
    
    return 0;
}
