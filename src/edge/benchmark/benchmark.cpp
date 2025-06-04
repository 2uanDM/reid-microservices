#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <string>
#include <omp.h>

class FlopsBenchmark {
private:
    static constexpr size_t MATRIX_SIZE = 1024;
    static constexpr int ITERATIONS = 10;
    
public:
    void printSystemInfo() {
        std::cout << "=== System Information ===" << std::endl;
        
        // Get CPU info
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("model name") != std::string::npos) {
                std::cout << "CPU: " << line.substr(line.find(":") + 2) << std::endl;
                break;
            }
        }
        
        std::cout << "CPU Cores: " << std::thread::hardware_concurrency() << std::endl;
        std::cout << "Matrix Size: " << MATRIX_SIZE << "x" << MATRIX_SIZE << std::endl;
        std::cout << "Iterations: " << ITERATIONS << std::endl;
        std::cout << std::endl;
    }
    
    double benchmarkSinglePrecision(bool multicore = false) {
        std::vector<std::vector<float>> A(MATRIX_SIZE, std::vector<float>(MATRIX_SIZE, 1.0f));
        std::vector<std::vector<float>> B(MATRIX_SIZE, std::vector<float>(MATRIX_SIZE, 2.0f));
        std::vector<std::vector<float>> C(MATRIX_SIZE, std::vector<float>(MATRIX_SIZE, 0.0f));
        
        if (multicore) {
            omp_set_num_threads(std::thread::hardware_concurrency());
        } else {
            omp_set_num_threads(1);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int iter = 0; iter < ITERATIONS; ++iter) {
            // Matrix multiplication with additional operations
            #pragma omp parallel for if(multicore) collapse(2)
            for (size_t i = 0; i < MATRIX_SIZE; ++i) {
                for (size_t j = 0; j < MATRIX_SIZE; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < MATRIX_SIZE; ++k) {
                        sum += A[i][k] * B[k][j];
                    }
                    C[i][j] = sum + std::sin(sum * 0.001f) + std::cos(sum * 0.001f);
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Calculate FLOPS
        // Matrix mult: N^3 * 2 operations (multiply + add)
        // Additional ops: N^2 * 3 operations (sin + cos + add) per iteration
        long long ops_per_iter = (long long)MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE * 2 + 
                                (long long)MATRIX_SIZE * MATRIX_SIZE * 3;
        long long total_ops = ops_per_iter * ITERATIONS;
        
        double seconds = duration.count() / 1000000.0;
        double gflops = (total_ops / seconds) / 1e9;
        
        return gflops;
    }
    
    double benchmarkDoublePrecision(bool multicore = false) {
        std::vector<std::vector<double>> A(MATRIX_SIZE, std::vector<double>(MATRIX_SIZE, 1.0));
        std::vector<std::vector<double>> B(MATRIX_SIZE, std::vector<double>(MATRIX_SIZE, 2.0));
        std::vector<std::vector<double>> C(MATRIX_SIZE, std::vector<double>(MATRIX_SIZE, 0.0));
        
        if (multicore) {
            omp_set_num_threads(std::thread::hardware_concurrency());
        } else {
            omp_set_num_threads(1);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int iter = 0; iter < ITERATIONS; ++iter) {
            #pragma omp parallel for if(multicore) collapse(2)
            for (size_t i = 0; i < MATRIX_SIZE; ++i) {
                for (size_t j = 0; j < MATRIX_SIZE; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < MATRIX_SIZE; ++k) {
                        sum += A[i][k] * B[k][j];
                    }
                    C[i][j] = sum + std::sin(sum * 0.001) + std::cos(sum * 0.001);
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        long long ops_per_iter = (long long)MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE * 2 + 
                                (long long)MATRIX_SIZE * MATRIX_SIZE * 3;
        long long total_ops = ops_per_iter * ITERATIONS;
        
        double seconds = duration.count() / 1000000.0;
        double gflops = (total_ops / seconds) / 1e9;
        
        return gflops;
    }
    
    void runBenchmark() {
        printSystemInfo();
        
        std::cout << "=== FLOPS Benchmark Results ===" << std::endl;
        
        // Single-core benchmarks
        std::cout << "\n--- Single Core Performance ---" << std::endl;
        std::cout << "Running Single Precision (FP32) benchmark..." << std::endl;
        double sp_gflops_single = benchmarkSinglePrecision(false);
        
        std::cout << "Running Double Precision (FP64) benchmark..." << std::endl;
        double dp_gflops_single = benchmarkDoublePrecision(false);
        
        // Multi-core benchmarks
        std::cout << "\n--- Multi Core Performance ---" << std::endl;
        std::cout << "Running Single Precision (FP32) benchmark..." << std::endl;
        double sp_gflops_multi = benchmarkSinglePrecision(true);
        
        std::cout << "Running Double Precision (FP64) benchmark..." << std::endl;
        double dp_gflops_multi = benchmarkDoublePrecision(true);
        
        // Results
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n=== Single Core Results ===" << std::endl;
        std::cout << "Single Precision: " << sp_gflops_single << " GFLOPS" << std::endl;
        std::cout << "Double Precision: " << dp_gflops_single << " GFLOPS" << std::endl;
        std::cout << "SP/DP Ratio: " << (sp_gflops_single / dp_gflops_single) << std::endl;
        
        std::cout << "\n=== Multi Core Results ===" << std::endl;
        std::cout << "Single Precision: " << sp_gflops_multi << " GFLOPS" << std::endl;
        std::cout << "Double Precision: " << dp_gflops_multi << " GFLOPS" << std::endl;
        std::cout << "SP/DP Ratio: " << (sp_gflops_multi / dp_gflops_multi) << std::endl;
        
        std::cout << "\n=== Scaling Efficiency ===" << std::endl;
        std::cout << "SP Speedup: " << (sp_gflops_multi / sp_gflops_single) << "x" << std::endl;
        std::cout << "DP Speedup: " << (dp_gflops_multi / dp_gflops_single) << "x" << std::endl;
        std::cout << "Theoretical Max: " << std::thread::hardware_concurrency() << "x" << std::endl;
    }
};

int main() {
    FlopsBenchmark benchmark;
    benchmark.runBenchmark();
    return 0;
}