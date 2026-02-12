#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

const int MAXBC = 128;

template <const int maxSCol>
__global__ void forward_kernel
    (const float* Q, const float* K, const float* V, const int N, const int d,
    const int Tc, const int Tr, const int Bc, const int Br, 
    const float softmax_scale, float* l, float *m, float* O) {
    // 注意此处 K 传递进来是没有转置的 K

    const int thidx = threadIdx.x;
    const int thnum = blockDim.x;
    const int cRow = blockIdx.y;    // batch_size
    const int cCol = blockIdx.x;    // num_head
    const int cColNum = gridDim.x;  // num of num_head

    // 动态共享内存 (Dynamic Shared Memory Allocation)
    // 在 CUDA 中，当你使用动态共享内存时，编译器在编译阶段并不知道你到底要用多少 Shared Memory。
    // 真正的内存分配指令是由 CUDA Runtime 在执行 kernel<<<grid, block, shared_mem_size>>> 
    // 这一行时发出的。(主要是由 shared_mem_size 指定)
    // 如果没有 extern，编译器会尝试在静态区分配空间，这要求你必须写死大小
    extern __shared__ float smem[];
    const int tile_size = Br * d;
    float* Qi = smem;
    float* Oi = &smem[tile_size];
    float* Kj = &smem[tile_size * 2];
    float* Vj = &smem[tile_size * 3];

    // 处理 Q, K, V, O 在 (batch_size, num_head, ) 的偏移
    // Q K V O shape == (batch_size, num_head, N, d)
    const int qkvo_offset = (cRow * cColNum * N * d) + (cCol * N * d);
    // 处理 l, m, 在 (batch_size, num_head, ) 的偏移
    // l m shape == (batch_size, num_head, N, 1) 
    const int lm_offset = (cRow * cColNum * N * 1) + (cCol * N * 1);

    for (int j = 0; j < Tc; j++) {
        // 加载 Kj, Vj to SMEM
        // 采用此方式加载减轻 bank conflict, 可进一步提升性能
        for (int idx = thidx; idx < tile_size; idx += thnum) {
            // 注意此处我们 K 没有进行转置
            Kj[idx] = K[qkvo_offset + j * Bc * d + idx];
            Vj[idx] = V[qkvo_offset + j * Bc * d + idx];
        }
        __syncthreads();

        for (int i = 0; i < Tr; i++) {
            // 加载 Qi, Oi to SMEM
            for (int idx = thidx; idx < tile_size; idx += thnum) {
                Qi[idx] = Q[qkvo_offset + i * Br * d + idx];
                Oi[idx] = O[qkvo_offset + i * Br * d + idx];
            }
            __syncthreads();

            // 加载 li, mi to Register
            // thread block 共 Br 个 thread, 每个 thread 负责 l, m 在Tiling其中一个值
            // i*Br*d 处理 Tiling偏移
            float thli = l[lm_offset + i * Br * 1 + thidx];
            float thmi = m[lm_offset + i * Br * 1 + thidx];
            
            // 计算 Sij = QK^T / sqrt(d) 并维护 行max
            // 每个thread负责 Sij 中一行数据
            float thSij[maxSCol];
            float thmij = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float val = 0;
                for (int x = 0; x < d; x++) {
                    // 注意此处我们 K 没有进行转置
                    val += Qi[thidx * d + x] * Kj[thidx * d + x];
                }
                val *= softmax_scale;
                thmij = fmaxf(thmij, val);

                thSij[y] = val;
            }
            
            // 计算 e^(QK^T / sqrt(d) - max) (即 flash attention 论文公式中的 Pij) 
            // 并维护 行sum
            float thlij = 0;
            for (int y = 0; y < Bc; y++) {
                thSij[y] = __expf(thSij[y] - thmij);
                thlij += thSij[y];
            }
            
            // 更新最新的 mij, lij
            float new_thmij = fmaxf(thmij, thmi); // 注意其中max值的计算为 QK^T / sqrt(d)
            float new_thlij = thli * __expf(thmi - new_thmij) + 
                        thlij * __expf(thmij - new_thmij);  // 注意其中sum值的计算为 sum(e^(QK^T / sqrt(d) - max))
            
            // 基于最新的 mij, lij 计算 PijVj 和 更新Oi 并将其累加更新最终的 Oi
            float oexpscale = __expf(thmi - new_thmij);
            float pvexpscale = __expf(thmij - new_thmij);
            for (int y = 0; y < d; y++) {
                float val = 0;
                for (int x = 0; x < Bc; x++) {
                    val += thSij[x] * Vj[x * d + y];
                }

                // 注意其中Oi值的计算为 e^(QK^T / sqrt(d) - max) / sum(e^(QK^T / sqrt(d) - max))
                // * thli 表示清除原先的 sum(e^(QK^T / sqrt(d) - max))
                // * expscale 表示更新 e^(QK^T / sqrt(d) - max) 中的 max
                // / new_thlij 表示除以新的 sum(e^(QK^T / sqrt(d) - max))
                Oi[thidx * d + y] = (Oi[thidx * d + y] * thli * oexpscale + 
                                     val * pvexpscale) / new_thlij;
                // Oi 在SMEM中，上述我访问Oi的方式每个thread间隔d取元素
                // 这样会带来较为严重的 bank Conflict
                // 优化方式只能考虑重写 如何组织 thread block 完成Tiling flash attention
            }
            
            // 写回 Oi
            for (int idx = thidx; idx < tile_size; idx += thnum) {
                O[qkvo_offset + i * Br * d + idx] = Oi[idx];
            }
            __syncthreads();
            // 写回 li, mi
            l[lm_offset + i * Br * 1 + thidx] = new_thlij;
            m[lm_offset + i * Br * 1 + thidx] = new_thmij;
        }

        // 此处__syncthreads();是必须的，否则其他线程还在使用 Kj, Vj 时
        // 提前完成的线程就更新掉 Kj, Vj 了
        __syncthreads();
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    
    const int sm_smem = prop.sharedMemPerMultiprocessor;
    const float use_ratio = 0.9;
    const int used_sm_smem = ceil(sm_smem * use_ratio);

    // --- Shared Memory (SRAM) ---
    // 这是单个 SM 上物理存在的最大 Shared Memory 总量
    std::cout << "Shared Memory per Multiprocessor (SM): " 
              << sm_smem / 1024.0 << " KB" << std::endl;
    std::cout << "We used " << use_ratio << "Shared Memory per Multiprocessor (SM): "
              << used_sm_smem / 1024.0 << " KB" << std::endl;

    const int batch_size = Q.size(0);
    const int num_head = Q.size(1);
    const int sequence_len = Q.size(2);
    const int head_dim = Q.size(3);

    // 为方便理解此处变量名和论文中对齐
    const int N = sequence_len;
    const int d = head_dim;

    // 此处实现 Flash Atteention论文中不太一样
    // 为代码实现方便直接令 Br == Bc, 且为了实现Sij compute on Chip(on Chip意味则Sij保存在thread Regiser中)
    // 我并不希望Register数据溢出到local mem中，因为这会导致性能严重下降
    // 而我的实现中每个thread处理 Tiling Sij 中的一行，其大小由 Bc 决定，所以 Bc 不能太大
    const int Bc = std::min(used_sm_smem / 4 / d, MAXBC);
    const int Br = Bc;
    
    const int Tr = ceil(N / Br);
    const int Tc = ceil(N / Bc);
    const float softmax_scale = 1.0 / sqrt(d);

    const int real_used_sm_smem = 4 * (Br * d) * sizeof(float);

    std::cout << "Paramters: \n"
              << " (N, d): " << "(" << N << "," << d << ")\n"
              << " Br: " << Br << "\n"
              << " Bc: " << Bc << "\n"
              << " Tr: " << Tr << "\n"
              << " Tc: " << Tc << std::endl;
    std::cout << "We really used Shared Memory per Multiprocessor (SM): "
              << real_used_sm_smem / 1024.0 << " KB" << std::endl;

    // 初始化 O, l, m to HBM
    // zeros_like 函数不仅仅拷贝形状（Shape）和数据类型（Dtype），
    // 它还会深度继承参考对象（即 $Q$）的所有配置，包括它所在的 Device。
    // $Q$ 通常已经作为输入存在于显卡上了。因此，zeros_like(Q) 创建出来的 $O$ 出生就在显卡上
    auto O = torch::zeros_like(Q);
    // 利用 Q.options() 自动对齐设备和数据类型。
    auto l = torch::zeros({batch_size, num_head, N}, Q.options());  // sum
    auto m = torch::full({batch_size, num_head, N}, 
             -std::numeric_limits<float>::infinity(), Q.options()); // max

    dim3 grid_dim(num_head, batch_size);
    dim3 block_dim(Br);

    // CUDA 默认将每个 Kernel 的 Shared Memory 限制在 48KB 以保证兼容性。 
    // 解决：你必须在 Kernel Launch 之前的 Host 代码中加入以下“动态配置”代码：
    cudaFuncSetAttribute(
        forward_kernel<MAXBC>, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        used_sm_smem
    );
    forward_kernel<MAXBC><<<grid_dim, block_dim, real_used_sm_smem>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    return O;
}