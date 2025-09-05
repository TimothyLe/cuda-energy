/* Author: Thinh Le
 * Date: September 2025
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Optimized shared memory matrix multiplication
__global__ void matrixMulShared(float* A, float* B, float* C, int N) {
    const int TILE_SIZE = 16;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && t * TILE_SIZE + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        // Compute partial result
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// CUDA kernel for signal processing (codec signal handling)
__global__ void processEnergySignals(float* rawSignals, float* processedSignals, 
                                   float* powerMatrix, int numSamples, int numChannels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y;
    
    if (idx < numSamples && channel < numChannels) {
        int globalIdx = channel * numSamples + idx;
        
        // Apply digital filtering (simple moving average for demo)
        float filtered = 0.0f;
        int windowSize = 5;
        int start = max(0, idx - windowSize/2);
        int end = min(numSamples - 1, idx + windowSize/2);
        
        for (int i = start; i <= end; ++i) {
            filtered += rawSignals[channel * numSamples + i];
        }
        filtered /= (end - start + 1);
        
        // Calculate instantaneous power (V*I for energy monitoring)
        float voltage = filtered;
        float current = rawSignals[globalIdx + numChannels * numSamples]; // Assume current in next block
        float power = voltage * current;
        
        processedSignals[globalIdx] = filtered;
        powerMatrix[globalIdx] = power;
    }
}

// FFT-based frequency domain analysis for codec optimization
__global__ void codecFrequencyAnalysis(cufftComplex* fftData, float* energySpectrum, 
                                     int numFreqBins, float sampleRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numFreqBins) {
        // Calculate power spectral density
        float real = fftData[idx].x;
        float imag = fftData[idx].y;
        float magnitude = sqrt(real * real + imag * imag);
        
        // Normalize and convert to dB for energy monitoring
        energySpectrum[idx] = 20.0f * log10f(magnitude + 1e-10f);
        
        // Detect energy spikes for server performance optimization
        float frequency = (float)idx * sampleRate / (2.0f * numFreqBins);
        if (energySpectrum[idx] > -20.0f && frequency > 50.0f && frequency < 1000.0f) {
            // Mark high-energy frequencies that may indicate server load
            energySpectrum[idx] += 10.0f; // Boost for attention
        }
    }
}

// Host function for energy monitoring system
class EnergyMonitorCUDA {
private:
    float *d_rawSignals, *d_processedSignals, *d_powerMatrix;
    float *d_matrixA, *d_matrixB, *d_matrixC;
    cufftComplex *d_fftData;
    float *d_energySpectrum;
    cufftHandle fftPlan;
    
    int numSamples;
    int numChannels;
    int matrixSize;

public:
    EnergyMonitorCUDA(int samples, int channels, int mSize) 
        : numSamples(samples), numChannels(channels), matrixSize(mSize) {
        
        // Allocate GPU memory
        cudaMalloc(&d_rawSignals, numSamples * numChannels * 2 * sizeof(float));
        cudaMalloc(&d_processedSignals, numSamples * numChannels * sizeof(float));
        cudaMalloc(&d_powerMatrix, numSamples * numChannels * sizeof(float));
        
        cudaMalloc(&d_matrixA, matrixSize * matrixSize * sizeof(float));
        cudaMalloc(&d_matrixB, matrixSize * matrixSize * sizeof(float));
        cudaMalloc(&d_matrixC, matrixSize * matrixSize * sizeof(float));
        
        cudaMalloc(&d_fftData, numSamples * sizeof(cufftComplex));
        cudaMalloc(&d_energySpectrum, numSamples * sizeof(float));
        
        // Create FFT plan
        cufftPlan1d(&fftPlan, numSamples, CUFFT_R2C, numChannels);
    }
    
    ~EnergyMonitorCUDA() {
        cudaFree(d_rawSignals);
        cudaFree(d_processedSignals);
        cudaFree(d_powerMatrix);
        cudaFree(d_matrixA);
        cudaFree(d_matrixB);
        cudaFree(d_matrixC);
        cudaFree(d_fftData);
        cudaFree(d_energySpectrum);
        cufftDestroy(fftPlan);
    }
    
    void processEnergyData(float* hostRawSignals, float* hostMatrixA, float* hostMatrixB) {
        // Copy input data to GPU
        cudaMemcpy(d_rawSignals, hostRawSignals, 
                   numSamples * numChannels * 2 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrixA, hostMatrixA, 
                   matrixSize * matrixSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrixB, hostMatrixB, 
                   matrixSize * matrixSize * sizeof(float), cudaMemcpyHostToDevice);
        
        // Launch signal processing kernel
        dim3 signalBlock(256);
        dim3 signalGrid((numSamples + signalBlock.x - 1) / signalBlock.x, numChannels);
        
        processEnergySignals<<<signalGrid, signalBlock>>>(
            d_rawSignals, d_processedSignals, d_powerMatrix, numSamples, numChannels);
        
        // Launch matrix multiplication for correlation analysis
        const int TILE_SIZE = 16;
        dim3 matrixBlock(TILE_SIZE, TILE_SIZE);
        dim3 matrixGrid((matrixSize + TILE_SIZE - 1) / TILE_SIZE, 
                       (matrixSize + TILE_SIZE - 1) / TILE_SIZE);
        
        matrixMulShared<<<matrixGrid, matrixBlock>>>(
            d_matrixA, d_matrixB, d_matrixC, matrixSize);
        
        // Perform FFT for frequency analysis
        cufftExecR2C(fftPlan, d_processedSignals, d_fftData);
        
        // Analyze frequency spectrum
        dim3 fftBlock(256);
        dim3 fftGrid((numSamples + fftBlock.x - 1) / fftBlock.x);
        
        codecFrequencyAnalysis<<<fftGrid, fftBlock>>>(
            d_fftData, d_energySpectrum, numSamples, 44100.0f);
        
        cudaDeviceSynchronize();
    }
    
    void getResults(float* hostProcessedSignals, float* hostMatrixC, float* hostEnergySpectrum) {
        cudaMemcpy(hostProcessedSignals, d_processedSignals, 
                   numSamples * numChannels * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostMatrixC, d_matrixC, 
                   matrixSize * matrixSize * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostEnergySpectrum, d_energySpectrum, 
                   numSamples * sizeof(float), cudaMemcpyDeviceToHost);
    }
};

// Example usage for server energy monitoring
int main() {
    const int NUM_SAMPLES = 4096;
    const int NUM_CHANNELS = 8;  // 8-channel energy monitoring
    const int MATRIX_SIZE = 512;
    
    // Initialize energy monitor
    EnergyMonitorCUDA monitor(NUM_SAMPLES, NUM_CHANNELS, MATRIX_SIZE);
    
    // Allocate host memory
    std::vector<float> rawSignals(NUM_SAMPLES * NUM_CHANNELS * 2);
    std::vector<float> processedSignals(NUM_SAMPLES * NUM_CHANNELS);
    std::vector<float> matrixA(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<float> matrixB(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<float> matrixC(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<float> energySpectrum(NUM_SAMPLES);
    
    // Generate sample energy monitoring data
    for (int i = 0; i < rawSignals.size(); ++i) {
        rawSignals[i] = sin(2.0f * M_PI * i / 100.0f) + 0.1f * sin(2.0f * M_PI * i / 10.0f);
    }
    
    // Initialize correlation matrices for pattern analysis
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
        matrixA[i] = (float)rand() / RAND_MAX;
        matrixB[i] = (float)rand() / RAND_MAX;
    }
    
    // Process energy data
    auto start = std::chrono::high_resolution_clock::now();
    monitor.processEnergyData(rawSignals.data(), matrixA.data(), matrixB.data());
    auto end = std::chrono::high_resolution_clock::now();
    
    // Get results
    monitor.getResults(processedSignals.data(), matrixC.data(), energySpectrum.data());
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "CUDA Energy Monitor Performance:" << std::endl;
    std::cout << "Processing time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Samples processed: " << NUM_SAMPLES * NUM_CHANNELS << std::endl;
    std::cout << "Matrix operations: " << MATRIX_SIZE << "x" << MATRIX_SIZE << std::endl;
    
    // Display sample results
    std::cout << "\nSample Energy Spectrum (first 10 bins):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "Bin " << i << ": " << energySpectrum[i] << " dB" << std::endl;
    }
    
    // Check for energy anomalies (server performance indicators)
    int anomalies = 0;
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        if (energySpectrum[i] > -10.0f) {  // High energy threshold
            anomalies++;
        }
    }
    
    std::cout << "\nServer Performance Indicators:" << std::endl;
    std::cout << "High-energy frequency bins detected: " << anomalies << std::endl;
    std::cout << "Energy monitoring efficiency: " << 
        (float)(NUM_SAMPLES * NUM_CHANNELS) / duration.count() << " samples/μs" << std::endl;
    
    return 0;
}
