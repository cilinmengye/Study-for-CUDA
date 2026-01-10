main.cu, warptiling.cuh, perf.cu 是基于 [aleksagordic代码](https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/10_kernel_warptiling.cuh)进行测试使用的代码，warptiling.cuh和aleksagordic的代码没有区别

original.ncu-rep是基于ncu工具对main.cu进行性能采集得到的文件，可用于Nsight Compute profiler进行可视化性能分析

<hr>

maintt.cu, warptiling_threadtiling.cuh, perftt.cu 是我想到另外一种warp处理子分块的方法，我实现了他。

tt_00.ncu-rep是在kernel参数如下时得到的性能分析文件：
    // 设置kernel参数
    const int BM = 128;
    const int BN = 128;
    const int BK = 16;
    const int WM = 64;
    const int WN = 64;
    const int TM = 16
    const int TN = 8;
    const int TK = 4;
    const int NUM_THREADS = 128;
通过分析发现其性能严重下降，是因为thread的寄存器不够了，例如float threadResults[TM * TN]这样较大的数组不是存放在寄存器中，而是被存放在了Locak Memory

tt.ncu-rep是在kernel参数如下时得到的性能分析文件：
    // 设置kernel参数
    const int BM = 128;
    const int BN = 128;
    const int BK = 16;
    const int WM = 64;
    const int WN = 64;
    const int TM = 16
    const int TN = 8;
    const int TK = 2;
    const int NUM_THREADS = 128;
这是为了解决上述thread的寄存器不够了的妥协才将TK设置为2, 通过分析其有严重的Bank Conflict问题
