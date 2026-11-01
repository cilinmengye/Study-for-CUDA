# Reference

mini flash attention v00 code 思路来自 [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal)

后续相关优化思路来自[【CUDA编程】flash attention CUDA 算子实现思路](https://zhuanlan.zhihu.com/p/1893608547121611896)

# 开发记录

## [2026.2.12] mini flash attention v00 完成。

**性能**

Device: NVIDIA GeForce RTX 4090
Compute Capability: 8.9
Shared Memory per Multiprocessor (SM): 100 KB
We used 0.9 Shared Memory per Multiprocessor (SM): 90 KB
Paramters: 
 (N, d): (1024,128)
 Br: 32
 Bc: 32
 Tr: 32
 Tc: 32
We really used Shared Memory per Multiprocessor (SM): 64 KB

pytorch manual attention
Self CPU time total: 90.702ms
Self CUDA time total: 91.218ms

cuda mini flash attention
Self CPU time total: 328.363ms
Self CUDA time total: 4.122ms

**缺陷**

目前 mini flash attention v00 仅支持 需要满足 N % Br == 0 && Br == Bc 的条件 且 MAXBC = 128。同时在代码中有一些严重的bank conflict。

* N % Br == 0 && Br == Bc 的条件考量在于代码简单好写，否则需要在代码中处理需要边界条件。
    * 例如 边界条件缺失。 当 N % Br != 0 时，最后一个 Tile 的线程会访问越界。在加载 Q, K, V 和写回 O 时，必须判断边界条件防止 Out of bound
    * Attention Mask 缺失。除了加载数据越界，计算 $S_{ij} = Q K^T$ 时，如果 K 的某些列超出了实际序列长度 $N$（在最后一个 Tile），这些位置计算出的 Attention Score 必须被 Mask 成 -inf，否则它们会参与 Softmax 计算，导致结果错误。

* MAXBC = 128 的条件考量在于 我利用thread block处理flash attention的方式。 在mini flash attention v00中，我一个thread block有 Br个线程，每个线程负责Sij的一行，因为Sij on chip完成计算，其值保存在寄存器上，而一个thread寄存器最多255个；每个线程负责Sij的一行, 即shape == (1, Bc)。所以Bc不能太大，否则装不下一个thread寄存器，会将多余数据存放在local mem中造成严重性能问题。
    * 解决方式考虑 思考其他thread block处理flash attention的方式；或者tensor parallelism（张量并行），将 model 大小 压力分散(我还没考虑清楚如何做)？

* 同时在代码中有一些严重的bank conflict。解决方法思考其他thread block处理flash attention的方式；
