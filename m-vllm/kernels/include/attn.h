
#include <torch/all.h>
#include <torch/library.h>
#include <torch/torch.h>




// void cutlass_mla_decode(
//     torch::Tensor const& out,
//     torch::Tensor const& q_nope,
//     torch::Tensor const& q_pe,
//     torch::Tensor const& kv_c_and_k_pe_cache,
//     torch::Tensor const& seq_lens,
//     torch::Tensor const& page_table,
//     torch::Tensor const& workspace,
//     double sm_scale,
//     int64_t num_kv_splits = 1 /* Set to 1 to avoid cuda_graph issue by default. */);

void rms_norm_kernel(
    torch::Tensor const& x);

void mha_kernel(float* q, float* k, float* v, float* output, int batch_size, int head_size, int num_heads);