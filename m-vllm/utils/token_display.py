"""
将 token ID 列表翻译成可读文字，便于调试推理结果。

用法:
  # 在代码里
  from m_vllm.utils.token_display import decode_to_text, format_tokens_for_display
  text = decode_to_text(tokenizer, token_ids)
  display_str = format_tokens_for_display(tokenizer, token_ids, max_show=50)

  # 命令行：从模型目录加载 tokenizer，解析 token 列表并展示
  python -m m_vllm.utils.token_display --model_path /path/to/Qwen3-4B --ids "9707, 11, 1879, 0, 5765, 79186"
  # 或粘贴整段 log（会从 seq.token_ids=[...] 里解析）
  python -m m_vllm.utils.token_display --model_path /path/to/Qwen3-4B --from_log "seq.token_ids=[9707, 11, ...], token_id=79186"
"""
import argparse
import re
import os
import sys

from typing import List, Union


def _ensure_tokenizer(model_path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, use_fast=True)


def decode_to_text(tokenizer, token_ids: List[int], skip_special_tokens: bool = True) -> str:
    """把 token ID 列表解码成整段文字。"""
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def format_tokens_for_display(
    tokenizer,
    token_ids: List[int],
    max_show: int = 80,
    sep: str = " | ",
    repr_empty: str = "<empty>",
) -> str:
    """
    把每个 token id 转成 id->'文字' 的形式，便于对照。
    max_show: 最多展示前多少个 token，超出部分用 ... (共 N 个) 概括。
    """
    if not token_ids:
        return repr_empty

    parts = []
    for i, tid in enumerate(token_ids[:max_show]):
        try:
            t = tokenizer.decode([tid], skip_special_tokens=False)
            # 换行/空格等转成可见
            t = repr(t)
        except Exception:
            t = f"<err:{tid}>"
        parts.append(f"{tid}->{t}")

    if len(token_ids) > max_show:
        parts.append(f"... (共 {len(token_ids)} 个)")
    return sep.join(parts)


def parse_ids_from_log(log_line: str) -> List[int]:
    """
    从日志里解析 token 列表。支持:
    - seq.token_ids=[9707, 11, 1879, ...]
    - token_ids=[9707, 11, ...]
    - 纯数字行: 9707, 11, 1879
    """
    # seq.token_ids=[...] 或 token_ids=[...]
    m = re.search(r"token_ids\s*=\s*\[([^\]]*)\]", log_line)
    if m:
        s = m.group(1).strip()
        if not s:
            return []
        return [int(x.strip()) for x in s.split(",")]

    # 纯逗号分隔数字
    s = log_line.strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    parts = [x.strip() for x in s.split(",") if x.strip()]
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            break
    return out


def main():
    parser = argparse.ArgumentParser(description="Token ID 转可读文字")
    parser.add_argument("--model_path", type=str, required=True, help="模型目录（含 tokenizer）")
    parser.add_argument("--ids", type=str, default=None, help="逗号分隔的 token id，如: 9707, 11, 1879")
    parser.add_argument("--from_log", type=str, default=None, help="从日志行解析 token_ids，可粘贴 seq.token_ids=[...] 整段")
    parser.add_argument("--max_show", type=int, default=80, help="逐 token 展示时最多显示前几个")
    parser.add_argument("--no_skip_special", action="store_true", help="解码时不过滤 special tokens")
    args = parser.parse_args()

    if args.ids is None and args.from_log is None:
        parser.error("请指定 --ids 或 --from_log")

    if args.ids:
        token_ids = [int(x.strip()) for x in args.ids.split(",") if x.strip()]
    else:
        token_ids = parse_ids_from_log(args.from_log)

    if not token_ids:
        print("未解析到任何 token id")
        return 1

    model_path = os.path.abspath(args.model_path)
    if not os.path.isdir(model_path):
        print(f"模型目录不存在: {model_path}")
        return 1

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    tokenizer = _ensure_tokenizer(model_path)

    skip_special = not args.no_skip_special
    full_text = decode_to_text(tokenizer, token_ids, skip_special_tokens=skip_special)
    display_str = format_tokens_for_display(tokenizer, token_ids, max_show=args.max_show)

    print("=== 整段解码 ===")
    print(full_text)
    print()
    print("=== 逐 token 展示 (id -> 文字) ===")
    print(display_str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
