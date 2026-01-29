#!/usr/bin/env python3
"""
排查「推理 token 组不成合理句子」时，检查模型计算是否正常。

用法（在项目根目录）:
  python -m m_vllm.scripts.debug_inference [--model_path ../Qwen3-4B] [--compare_hf]

1. 单步 prefill：固定 prompt，跑一次 forward，打印 logits 统计和 argmax 对应 token。
2. 可选：与 HuggingFace 同模型、同输入对比 next-token logits 或 next-token id。
"""
import argparse
import os
import sys

# 保证能 import m_vllm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from transformers import AutoConfig, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Debug inference: check logits and next token")
    parser.add_argument("--model_path", type=str, default="../Qwen3-4B/", help="Model directory (safetensors + config)")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Prompt for sanity check")
    parser.add_argument("--compare_hf", action="store_true", help="Compare next-token logits with HuggingFace model")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model_path)
    if not os.path.isdir(model_path):
        print(f"Error: model_path is not a directory: {model_path}")
        sys.exit(1)

    from m_vllm.data_classes.config import GlobalConfig, ModelConfig
    from m_vllm.engine.model_runner import ModelRunner
    from m_vllm.models.modeloader import load_model

    # 使用与 engine 一致的 config
    model_config = ModelConfig(model_path=model_path, qwen3_config=AutoConfig.from_pretrained(model_path))
    global_config = GlobalConfig(
        model_name=model_path,
        model_config=model_config,
        eos=151643,  # 占位，仅 debug 用
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    if isinstance(token_ids, list):
        pass
    else:
        token_ids = token_ids.tolist()
    print(f"Prompt: {args.prompt!r}")
    print(f"Token ids (first 20): {token_ids[:20]} ... len={len(token_ids)}")

    # 我们的模型：单步 prefill（整段 prompt 进一次 forward，只看最后一个位置的 logits）
    model_runner = ModelRunner(model_path, global_config)
    load_model(model_runner.model, model_path)
    model_runner.model.eval()

    seq_len = len(token_ids)
    input_ids = torch.tensor([token_ids], dtype=torch.int64).cuda()  # (1, seq_len)
    positions = torch.arange(seq_len, dtype=torch.int64).cuda().unsqueeze(0)  # (1, seq_len)

    with torch.no_grad():
        logits = model_runner.run(input_ids, positions)  # (1, seq_len, vocab_size) 或 (1*seq_len, vocab_size)

    # 兼容不同 shape：若为 (1, seq_len, V) 取 [0, -1]；若为 (seq_len, V) 取 [-1]
    if logits.dim() == 3:
        last_logits = logits[0, -1, :].float()
    else:
        last_logits = logits[-1, :].float()

    print("\n--- Our model (last position logits) ---")
    print(f"Shape: {last_logits.shape}, dtype: {last_logits.dtype}")
    print(f"Min: {last_logits.min().item():.4f}, Max: {last_logits.max().item():.4f}, Mean: {last_logits.mean().item():.4f}")
    print(f"Has NaN: {torch.isnan(last_logits).any().item()}, Has Inf: {torch.isinf(last_logits).any().item()}")

    pred_id = last_logits.argmax(dim=-1).item()
    pred_token = tokenizer.decode([pred_id])
    print(f"Argmax token id: {pred_id}, decoded: {pred_token!r}")

    if args.compare_hf:
        try:
            from transformers import Qwen3ForCausalLM as HFQwen3
        except Exception:
            print("HuggingFace Qwen3 not available, skip --compare_hf")
        else:
            print("\n--- HuggingFace same model (greedy next token) ---")
            hf_model = HFQwen3.from_pretrained(model_path, torch_dtype=torch.float32).eval().cuda()
            hf_input = torch.tensor([token_ids], dtype=torch.int64).cuda()
            with torch.no_grad():
                hf_out = hf_model(hf_input)
                hf_logits = hf_out.logits[0, -1, :].float()
            hf_pred_id = hf_logits.argmax(dim=-1).item()
            hf_pred_token = tokenizer.decode([hf_pred_id])
            print(f"HF Argmax token id: {hf_pred_id}, decoded: {hf_pred_token!r}")

            if pred_id == hf_pred_id:
                print("Match: our next token == HF next token.")
            else:
                print(f"Mismatch: our {pred_id} ({pred_token!r}) vs HF {hf_pred_id} ({hf_pred_token!r})")
            # 可选：diff logits（看是否整体一致）
            diff = (last_logits - hf_logits).abs()
            print(f"Logits L1 diff (our vs HF): {diff.sum().item():.6f}, max diff: {diff.max().item():.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
