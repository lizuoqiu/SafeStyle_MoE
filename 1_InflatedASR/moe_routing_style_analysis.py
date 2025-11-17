#!/usr/bin/env python
import os
import json
import argparse
from typing import Dict, List, Any

import numpy as np
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# 你之前定义过的 12 种文风
STYLE_KEYS: List[str] = [
    "shakespeare",
    "biblical",
    "opera",
    "epic_poetry",
    "noir",
    "high_fantasy",
    "cyberpunk",
    "scientific",
    "news",
    "bedtime",
    "philosophical",
    "mythological",
]

STYLE_TO_COLUMN: Dict[str, str] = {
    style: f"{style} Query" for style in STYLE_KEYS
}


# --------------------- 工具函数：JS divergence & routing 收集 --------------------- #

def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """Jensen-Shannon divergence between two probability vectors."""
    p = p.astype(np.float64)
    q = q.astype(np.float64)

    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return float((a * np.log(a / b)).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def collect_router_means_for_texts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int = 8,
    max_length: int = 512,
    desc: str = "",
) -> np.ndarray:
    """
    对一组文本，收集每个样本的 MoE routing 平均分布。

    返回:
        probs: np.ndarray，shape = (N, L, E)
            N = 样本数
            L = 有效的 MoE 层数
            E = 每层专家数

        probs[n, l, :] 是第 n 个样本在第 l 层上的平均专家分布。
    """
    device = next(model.parameters()).device
    model.eval()

    all_batches = []

    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[start:start + batch_size]
        batch_texts = [t if isinstance(t, str) else "" for t in batch_texts]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

        B, T_base = input_ids.shape  # 这里 B=8, T_base=19
        mask_base = attention_mask.unsqueeze(-1)  # (B, T_base, 1)

        with torch.inference_mode():
            outputs = model(
                **enc,
                output_router_logits=True,
                use_cache=False,
                return_dict=True,
            )

        router_logits = outputs.router_logits

        # 统一成 tuple/list
        if isinstance(router_logits, torch.Tensor):
            router_logits = (router_logits,)

        layer_means = []

        for layer_idx, logits_l in enumerate(router_logits):
            if logits_l is None:
                print(f"[Warn] router_logits[{layer_idx}] is None, skip this layer.")
                continue

            # 情况一：标准 3D，(B, T_router, E)
            if logits_l.dim() == 3:
                B_l, T_router, E = logits_l.shape

                gate_probs = torch.softmax(logits_l, dim=-1)  # (B_l, T_router, E)

                if T_router == T_base:
                    mask_l = mask_base
                else:
                    print(
                        f"[Warn] layer {layer_idx}: T_router={T_router} != T_base={T_base}, "
                        "using all-one mask for this layer."
                    )
                    mask_l = torch.ones(
                        (B_l, T_router, 1),
                        device=gate_probs.device,
                        dtype=gate_probs.dtype,
                    )

                gate_probs = gate_probs * mask_l
                token_counts = mask_l.sum(dim=1).clamp(min=1.0)  # (B_l, 1)
                mean_gate = gate_probs.sum(dim=1) / token_counts  # (B_l, E)

            # 情况二：2D，(D, E)
            elif logits_l.dim() == 2:
                D, E = logits_l.shape

                # 最常见的 OLMoE 情况：D = B * T_base = 152
                if D == B * T_base:
                    # 先 reshape 回 (B, T_base, E)
                    logits_3d = logits_l.view(B, T_base, E)
                    gate_probs = torch.softmax(logits_3d, dim=-1)  # (B, T_base, E)

                    mask_l = mask_base  # (B, T_base, 1)
                    gate_probs = gate_probs * mask_l
                    token_counts = mask_l.sum(dim=1).clamp(min=1.0)  # (B, 1)
                    mean_gate = gate_probs.sum(dim=1) / token_counts  # (B, E)

                # 已经是 (B, E) 的情况
                elif D == B:
                    gate_probs = torch.softmax(logits_l, dim=-1)  # (B, E)
                    mean_gate = gate_probs

                else:
                    # 兜底：未知形状，只能做 global mean
                    print(
                        f"[Warn] layer {layer_idx}: router_logits dim=2, shape={logits_l.shape}, "
                        f"cannot map D={D} to B={B} or B*T={B*T_base}. "
                        "Averaging over dim0 and broadcasting."
                    )
                    gate_probs = torch.softmax(logits_l, dim=-1)  # (D, E)
                    mean_single = gate_probs.mean(dim=0, keepdim=True)  # (1, E)
                    mean_gate = mean_single.repeat(B, 1)  # (B, E)

            else:
                # 更怪的情况：flatten 再平均
                print(
                    f"[Warn] layer {layer_idx}: unexpected router_logits dim={logits_l.dim()}, "
                    f"shape={logits_l.shape}. Flattening then averaging."
                )
                last_dim = logits_l.shape[-1]
                gate_probs = torch.softmax(
                    logits_l.view(-1, last_dim), dim=-1
                )  # (D_flat, E)
                mean_single = gate_probs.mean(dim=0, keepdim=True)  # (1, E)
                mean_gate = mean_single.repeat(B, 1)  # (B, E)

            layer_means.append(mean_gate.cpu().numpy())

        if not layer_means:
            raise RuntimeError("No valid router_logits layers were collected.")

        # layer_means: list 长度 L，每个元素 (B, E)
        batch_arr = np.stack(layer_means, axis=0).transpose(1, 0, 2)  # (B, L, E)
        all_batches.append(batch_arr)

    probs = np.concatenate(all_batches, axis=0)  # (N, L, E)
    return probs



def compute_routing_metrics(
    base_probs: np.ndarray,
    style_probs: np.ndarray,
    top_k: int = 2,
) -> Dict[str, Any]:
    """
    对某一 style，和 base (原始 query) 的 routing 做比较。

    输入:
        base_probs, style_probs: shape = (N, L, E)

    输出:
        {
            "mean_js": float,
            "mean_topk_overlap": float,
            "per_layer_js": List[float] (len = L),
            "per_layer_topk_overlap": List[float] (len = L),
        }
    """
    assert base_probs.shape == style_probs.shape
    N, L, E = base_probs.shape

    js_per_layer: List[float] = []
    overlap_per_layer: List[float] = []

    for l in range(L):
        js_vals = []
        overlap_vals = []

        for i in range(N):
            p = base_probs[i, l]
            q = style_probs[i, l]

            # JS divergence
            js_vals.append(js_divergence(p, q))

            # top-k expert overlap
            top_p = np.argsort(p)[-top_k:]
            top_q = np.argsort(q)[-top_k:]
            inter = len(set(top_p) & set(top_q))
            overlap_vals.append(inter / float(top_k))

        js_per_layer.append(float(np.mean(js_vals)))
        overlap_per_layer.append(float(np.mean(overlap_vals)))

    metrics = {
        "mean_js": float(np.mean(js_per_layer)),
        "mean_topk_overlap": float(np.mean(overlap_per_layer)),
        "per_layer_js": js_per_layer,
        "per_layer_topk_overlap": overlap_per_layer,
    }
    return metrics


# --------------------- 主流程 --------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Check whether literary style inputs perturb MoE routing."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMoE-1B-7B-0924-Instruct",
        help="MoE model name (must support output_router_logits).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="Data/Input/jailbreaks_literary_short_prompt.csv",
        help="CSV with 'Original Query' + '<style> Query' columns.",
    )
    parser.add_argument(
        "--base_column",
        type=str,
        default="Original Query",
        help="Column name for original jailbreak queries.",
    )
    parser.add_argument(
        "--styles",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Which style names to evaluate (subset of: %s). "
            "If not provided, use all."
        ) % ", ".join(STYLE_KEYS),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for routing analysis.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max token length for tokenization.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="Top-k experts for overlap metric.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="Data/Output/moe_routing_style_analysis.json",
        help="Where to save the aggregated metrics.",
    )
    args = parser.parse_args()

    print(f"[Info] Loading model: {args.model}")
    print(f"[Info] CSV file: {args.csv}")

    # 单卡版本：直接全 load 到一块 GPU 上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    # 关键：打开 router logits 输出
    model.config.output_router_logits = True

    # 读取 CSV
    df = pd.read_csv(args.csv)
    if args.base_column not in df.columns:
        raise ValueError(f"Base column '{args.base_column}' not found in CSV.")

    base_texts = df[args.base_column].fillna("").tolist()
    print(f"[Info] Number of samples: {len(base_texts)}")

    # 选择要评估的 styles
    if args.styles is None:
        styles_to_run = STYLE_KEYS
    else:
        # 用户给了一部分 style，就只跑那几个
        for s in args.styles:
            if s not in STYLE_KEYS:
                raise ValueError(
                    f"Unknown style '{s}'. Must be one of: {STYLE_KEYS}"
                )
        styles_to_run = args.styles

    # 先对 base（原始 query）收集 routing 分布
    print("[Info] Collecting routing for base (Original Query)...")
    base_probs = collect_router_means_for_texts(
        model=model,
        tokenizer=tokenizer,
        texts=base_texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        desc="Base queries",
    )

    results: Dict[str, Any] = {
        "model": args.model,
        "base_column": args.base_column,
        "num_samples": len(base_texts),
        "styles": {},
    }

    # 对每种风格做对比
    for style in styles_to_run:
        col_name = STYLE_TO_COLUMN.get(style, f"{style} Query")
        if col_name not in df.columns:
            print(f"[Warn] Column '{col_name}' not found, skip style '{style}'.")
            continue

        style_texts = df[col_name].fillna("").tolist()
        if len(style_texts) != len(base_texts):
            raise ValueError(
                f"Length mismatch between base ({len(base_texts)}) "
                f"and column '{col_name}' ({len(style_texts)})."
            )

        print(f"[Info] Collecting routing for style '{style}' (column: {col_name})...")
        style_probs = collect_router_means_for_texts(
            model=model,
            tokenizer=tokenizer,
            texts=style_texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            desc=f"Style: {style}",
        )

        metrics = compute_routing_metrics(
            base_probs=base_probs,
            style_probs=style_probs,
            top_k=args.top_k,
        )
        results["styles"][style] = {
            "column": col_name,
            **metrics,
        }

        print(
            f"[Summary] Style '{style}': "
            f"mean JS = {metrics['mean_js']:.4f}, "
            f"mean top-{args.top_k} overlap = {metrics['mean_topk_overlap']:.4f}"
        )

    # 保存结果
    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[Info] Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
