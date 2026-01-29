# m-vllm vs nano-vllm Qwen3 差异与修正说明

以 nano-vllm 为参考，对 m-vllm 的 Qwen3 做了以下修正。

---

## 1. 已修正（影响计算/加载）

### 1.1 Qwen3Attention：q_norm / k_norm 仅当无 qkv_bias 时使用

- **nano**：`if not self.qkv_bias:` 才创建并应用 `q_norm`、`k_norm`。
- **m-vllm 原问题**：始终创建并应用 q/k norm；当 `attention_bias=True` 时与 Qwen3 约定不符。
- **修正**：与 nano 一致，仅在 `not qkv_bias` 时创建并应用 `q_norm`、`k_norm`。

### 1.2 tie_word_embeddings

- **nano**：`if config.tie_word_embeddings: self.lm_head.weight.data = self.model.embed_tokens.weight.data`
- **m-vllm 原问题**：未处理，当 config 中 `tie_word_embeddings=True` 时 lm_head 与 embed 未共享。
- **修正**：在 `Qwen3ForCausalLM.__init__` 中增加相同逻辑（用 `getattr(..., False)` 安全读取）。

### 1.3 Qwen3ForCausalLM.forward 参数顺序

- **nano**：`forward(input_ids, positions)`，与 HF 习惯一致。
- **m-vllm 原问题**：`forward(positions, input_ids)`，易与调用方搞反。
- **修正**：改为 `forward(input_ids, positions)`；`model_runner.run` 中改为 `self.model(input_ids, positions)`。

### 1.4 compute_logits

- **nano**：直接 `return self.lm_head(hidden_states)`。
- **m-vllm 原问题**：中间变量 + `logger.info`，多余且可能影响性能。
- **修正**：改为直接 `return self.lm_head(hidden_states)`。

### 1.5 attention_bias 默认值

- **nano**：`qkv_bias=getattr(config, 'attention_bias', True)`。
- **m-vllm 原问题**：直接使用 `model_config.attention_bias`，config 缺字段时会报错。
- **修正**：改为 `getattr(model_config, "attention_bias", True)`。

---

## 2. 模型加载（packed_modules_mapping）

两边 `packed_modules_mapping` 一致，loader 逻辑等价：

- `q_proj` → `qkv_proj` + shard `"q"`
- `k_proj` → `qkv_proj` + shard `"k"`
- `v_proj` → `qkv_proj` + shard `"v"`
- `gate_proj` → `gate_up_proj` + 0
- `up_proj` → `gate_up_proj` + 1

m-vllm 的 `modeloader.load_model` 与 nano 的 `loader.load_model` 行为一致（按 packed 映射 + weight_loader 分片）。

---

## 3. 未改动的合理差异

- **DecoderLayer.forward 参数顺序**：nano 为 `self_attn(positions, hidden_states)`，m-vllm 为 `self_attn(hidden_states, position_ids)`，均为内部约定，与各自 Attention.forward 一致即可。
- **RoPE**：nano 的 `get_rope` 支持 `rotary_dim`、`rope_scaling` 等；m-vllm 当前仅 `(head_dim, max_position_embeddings, rope_theta)`。若需 long context 的 rope_scaling，需在 m-vllm 的 rope 层和 config 中再对齐。
- **TP**：nano 在 Attention 里用 `dist.get_world_size()` 做 head 切分；m-vllm 若单卡或 TP 在别处处理，可保持现状。

---

## 4. Layers 模型加载与计算（已修正）

以 nano-vllm 的 `layers/linear.py`、`layers/embed_head.py` 为参考，对 m-vllm 的 layers 做了对齐。

### 4.1 MergedColumnParallelLinear（gate_up_proj 等）

- **nano**：`__init__` 里先 `self.output_sizes = output_sizes`，再 `super().__init__(...)`，`weight_loader` 用 `self.output_sizes[:loaded_shard_id]` 等计算 shard_offset / shard_size。
- **m-vllm 原问题**：未保存 `output_size`，`weight_loader` 里使用 `self.output_size` 会 `AttributeError`，导致 gate_proj/up_proj 加载失败。
- **修正**：在 `super()` 前增加 `self.output_size = output_size`（与 nano 的 output_sizes 语义一致）。

### 4.2 RowParallelLinear（o_proj、down_proj 等）

- **nano**：`F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)`，仅 rank 0 加 bias。
- **m-vllm 原问题**：`F.linear(x, self.weight, self.bias)`，每 rank 都加 bias，all_reduce 后 bias 被加 tp_size 次，结果错误。
- **修正**：改为 `self.bias if self.tp_rank == 0 else None`，与 nano 一致。

### 4.3 ParallelLMHead.forward（gather logits）

- **nano**：`if self.tp_size > 1:` 时做 gather。
- **m-vllm 原问题**：误写为 `if self.tp_rank > 1:`，多卡时只有 rank 2+ 才 gather，rank 0/1 行为错误。
- **修正**：改为 `if self.tp_size > 1:`。

### 4.4 其他 layers 加载

- **RMSNorm**：两边均无自定义 `weight_loader`，使用 loader 的 `default_weight_loader` 全量 copy，行为一致。
- **VocabParallelEmbedding / Embed**：m-vllm 的 `weight_loader` 按 `tp_rank * vocab_size_p` 做 narrow 后 copy，与 nano 的按 partition 切分逻辑一致。
- **QKVParallelLinear**：两边均按 shard_id `"q"`/`"k"`/`"v"` 计算 offset/size 并 chunk 后 copy，逻辑一致。

---

## 5. 与 nano 流程对齐（decode 重复 token 根因）

### 5.1 未加载权重（根因）

- **nano**：`ModelRunner.__init__` 里 `load_model(self.model, config.model)`，模型从 safetensors 加载。
- **m-vllm 原问题**：`TPWorker.init_dist_and_load()` 只创建 `ModelRunner` 和 `allocate_kv_cache()`，**从未调用 `load_model`**，模型权重一直是随机初始化，decode 输出会乱/重复。
- **修正**：在 `init_dist_and_load()` 中，创建 `model_runner` 后立即调用 `load_model(self.model_runner.model, self.model_config.model_path)`。

### 5.2 每步后清空 context

- **nano**：`run()` 末尾调用 `reset_context()`，避免下一步用到上一步的 context。
- **m-vllm 原问题**：未调用 `reset_context()`。
- **修正**：在 `post_process` 之后、`send_result_to_engine` 之前调用 `reset_context()`。

---

## 6. 修改文件一览

| 文件 | 修改内容 |
|------|----------|
| `m_vllm/models/Qwen3.py` | q/k norm 条件化、tie_word_embeddings、forward(input_ids, positions)、compute_logits 精简、attention_bias getattr |
| `m_vllm/engine/model_runner.py` | 调用改为 `self.model(input_ids, positions)` |
| `m_vllm/engine/tp_worker.py` | **init_dist_and_load 中增加 load_model**；每步后 **reset_context()** |
| `m_vllm/layers/linear.py` | MergedColumnParallelLinear 增加 `self.output_size`；RowParallelLinear 仅 rank 0 加 bias |
| `m_vllm/layers/embed.py` | ParallelLMHead.forward 中 `tp_rank > 1` 改为 `tp_size > 1` |
