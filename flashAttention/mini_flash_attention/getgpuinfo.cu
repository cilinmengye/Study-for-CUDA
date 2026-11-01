#include <cuda_runtime.h>
#include <iostream>

void getSMResources() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // --- 1. Shared Memory (SRAM) ---
    // 这是单个 SM 上物理存在的最大 Shared Memory 总量
    std::cout << "Shared Memory per Multiprocessor (SM): " 
              << prop.sharedMemPerMultiprocessor / 1024.0 << " KB" << std::endl;

    // 这是单个 Thread Block 允许使用的最大 Shared Memory
    // 注意：默认情况下，这个值通常是 48KB。如果你需要更多（如 A100 支持 160KB+），
    // 你必须在 Host 代码中显式调用 cudaFuncSetAttribute 开启。
    std::cout << "Max Shared Memory per Block (Default/Opt-in): " 
              << prop.sharedMemPerBlock / 1024.0 << " KB" 
              << " (Can be higher on Ampere+ with cudaFuncSetAttribute)" << std::endl;

    // --- 2. Registers ---
    // 单个 SM 上的 32-bit 寄存器总数
    std::cout << "Registers per Multiprocessor (SM): " 
              << prop.regsPerMultiprocessor << " (Total 32-bit regs)" << std::endl;
    
    // 换算成 KB
    double regFileKB = (double)prop.regsPerMultiprocessor * 4.0 / 1024.0;
    std::cout << "Register File Size per SM (KB): " << regFileKB << " KB" << std::endl;

    // 单个 Block 允许使用的最大寄存器数
    std::cout << "Max Registers per Block: " << prop.regsPerBlock << std::endl;
}

int main() {
    getSMResources();
    return 0;
}