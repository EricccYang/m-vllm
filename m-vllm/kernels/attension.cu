#include <cuda_runtime.h>
#include "include/attn.h"


__global__ void flash_attention_kernel(float* q, float* k, float* v, float* output, int batch_size, int head_size, int num_heads) {
    int batch_idx = blockIdx.x;
    int head_idx = threadIdx.x;
    int idx = batch_idx * head_size + head_idx;
    output[idx] = q[idx] * k[idx] * v[idx];
}



// void use_kernel_function(float* q, float* k, float* v, float* output){


//     blockDim

// }


void rms_norm_kernel(torch::Tensor const& x){
    flash_attention_kernel<<<1, 1>>>(x.data_ptr<float>(), x.data_ptr<float>(), x.data_ptr<float>(), x.data_ptr<float>(), 1, 1, 1);
    return;
}