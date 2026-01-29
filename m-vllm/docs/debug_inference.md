# 推理 token 组不成合理句子：如何排查模型计算

## 1. 已修复的问题（请确认已合入）

- **model_runner 与 Qwen3ForCausalLM 参数顺序**  
  `Qwen3ForCausalLM.forward(positions, input_ids)` 要求先 `positions` 再 `input_ids`。  
  若误写为 `self.model(input_ids, positions)`，会把 **position 当 token、token 当 position** 喂进模型，输出会完全错乱。  
  正确写法：`self.model(positions, input_ids)`（见 `engine/model_runner.py`）。

## 2. 用脚本做「单步 prefill + 可选 HF 对比」

在项目根目录执行：

```bash
# 仅跑我们模型：看 logits 是否正常、argmax 下一个 token
python -m m_vllm.scripts.debug_inference --model_path /path/to/Qwen3-4B

# 与 HuggingFace 同模型对比下一个 token 是否一致
python -m m_vllm.scripts.debug_inference --model_path /path/to/Qwen3-4B --compare_hf
```

关注：

- logits 是否有 NaN/Inf、数值范围是否合理；
- 我们模型的 argmax next-token 与 HF 是否一致（一致则单步计算基本正确）。

## 3. 排查清单（按顺序）

| 步骤 | 检查项 | 说明 |
|------|--------|------|
| 1 | **forward 参数顺序** | `model_runner.run` 内调用 `self.model(positions, input_ids)`，不能反。 |
| 2 | **权重是否加载** | 确认 `load_model(model, path)` 对当前模型执行过，且 path 下存在对应 safetensors。 |
| 3 | **Prefill input_ids / positions** | prefill 时 `input_ids` 为整段 prompt 的 token id，`positions` 为 `[0,1,...,seq_len-1]`，且与 `input_ids` 一一对应。 |
| 4 | **Decode 的 last token** | decode 时 `input_ids` 应为「当前步要预测的上一个 token」即 `seq.last_token`，`positions` 为该序列当前长度减 1。 |
| 5 | **KV cache 与 slot_mapping** | prefill 写 cache、decode 读写的 slot 与 block_tables 是否与当前序列、position 一致；错位会导致后续步看到错误历史。 |
| 6 | **RoPE / position 编码** | 确认 attention 里用的 `positions` 与当前 token 在序列中的位置一致（含 paged KV 的 slot 对应关系）。 |
| 7 | **logits → token** | 采样前确认 logits 的最后一维是 vocab；若用 temperature，确认在 logits 上正确除温再取 argmax/sample。 |
| 8 | **token_id 类型** | `post_process` 里 append 到 `seq.token_ids` 的应为 Python `int`（对 tensor 用 `.item()` 或 `int(...)`），避免后续比较/解码出错。 |

## 4. 快速自检：固定 prompt 的贪婪解码

选一句短 prompt（如 `"The capital of France is"`），用我们的引擎做 **贪婪解码**（temperature=0 或等价 argmax）若干步，看生成是否与 HuggingFace 同模型、同 prompt、贪婪解码结果一致。若单步一致但多步不一致，重点查 **KV cache 与 decode 时的 positions/slot_mapping**。

## 5. 日志建议

在以下位置临时加 log 便于定位：

- `model_runner.run`：`input_ids.shape`、`positions.shape`、`logits.shape`（以及是否含 NaN）；
- prefill/decode 分支：每个 batch 的 `input_ids` 前几个 token、对应 `positions` 前几维；
- `post_process`：每条序列新 append 的 `token_id`（以及类型是否为 int）。

以上步骤能系统判断问题是在「单步前向」「权重/配置」「KV/position」还是「采样/后处理」环节。
