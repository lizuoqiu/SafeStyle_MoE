#!/usr/bin/env python
"""
可视化 MoE routing 指标的脚本。

输入：之前分析脚本生成的 JSON 文件（包含 per_layer_js, per_layer_topk_overlap 等）。
输出：多种 PNG 图像：
  - per_layer_js_lines.png
  - per_layer_js_heatmap.png
  - per_layer_overlap_heatmap.png
  - style_js_vs_overlap_scatter.png
"""

import os
import json
import argparse
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt


# ---------- 工具函数 ---------- #

def load_metrics(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ---------- 图 1：每种 style 在每一层的 JS 曲线 ---------- #

def plot_per_layer_js_lines(metrics: Dict[str, Any], out_dir: str) -> None:
    styles_dict = metrics["styles"]
    # 稳定排序一下，图里的图例顺序就固定了
    styles: List[str] = sorted(styles_dict.keys())

    # 假设所有 style 的层数相同，从第一个取
    num_layers = len(next(iter(styles_dict.values()))["per_layer_js"])
    x = np.arange(num_layers)

    plt.figure(figsize=(10, 6))

    for style in styles:
        per_layer_js = styles_dict[style]["per_layer_js"]
        plt.plot(x, per_layer_js, label=style)

    plt.xlabel("MoE layer index")
    plt.ylabel("JS divergence (routing)")
    plt.title(
        f"Per-layer JS divergence between base and style inputs\n"
        f"Model: {metrics.get('model', '')}"
    )
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "per_layer_js_lines.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Info] Saved line plot to {out_path}")


# ---------- 图 2：JS 热力图（style × layer） ---------- #

def plot_per_layer_js_heatmap(metrics: Dict[str, Any], out_dir: str) -> None:
    styles_dict = metrics["styles"]
    styles: List[str] = sorted(styles_dict.keys())

    num_layers = len(next(iter(styles_dict.values()))["per_layer_js"])
    js_matrix = np.zeros((len(styles), num_layers), dtype=float)

    for i, style in enumerate(styles):
        js_matrix[i] = np.array(styles_dict[style]["per_layer_js"], dtype=float)

    plt.figure(figsize=(10, 6))
    im = plt.imshow(js_matrix, aspect="auto")
    plt.colorbar(im, label="JS divergence")

    plt.yticks(range(len(styles)), styles)
    plt.xticks(range(num_layers), range(num_layers))
    plt.xlabel("MoE layer index")
    plt.ylabel("Style")
    plt.title("Layer-wise routing JS divergence (style vs base)")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "per_layer_js_heatmap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Info] Saved JS heatmap to {out_path}")


# ---------- 图 3：Top-k overlap 热力图（style × layer） ---------- #

def plot_per_layer_overlap_heatmap(metrics: Dict[str, Any], out_dir: str) -> None:
    styles_dict = metrics["styles"]
    styles: List[str] = sorted(styles_dict.keys())

    num_layers = len(next(iter(styles_dict.values()))["per_layer_topk_overlap"])
    overlap_matrix = np.zeros((len(styles), num_layers), dtype=float)

    for i, style in enumerate(styles):
        overlap_matrix[i] = np.array(
            styles_dict[style]["per_layer_topk_overlap"], dtype=float
        )

    plt.figure(figsize=(10, 6))
    im = plt.imshow(overlap_matrix, aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(im, label="Top-k expert overlap")

    plt.yticks(range(len(styles)), styles)
    plt.xticks(range(num_layers), range(num_layers))
    plt.xlabel("MoE layer index")
    plt.ylabel("Style")
    plt.title("Layer-wise top-k expert overlap (style vs base)")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "per_layer_overlap_heatmap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Info] Saved overlap heatmap to {out_path}")


# ---------- 图 4：Style 级别散点图（mean JS vs mean overlap） ---------- #

def plot_style_js_vs_overlap_scatter(metrics: Dict[str, Any], out_dir: str) -> None:
    styles_dict = metrics["styles"]
    styles: List[str] = sorted(styles_dict.keys())

    mean_js_list = []
    mean_overlap_list = []

    for style in styles:
        s_res = styles_dict[style]
        mean_js_list.append(float(s_res["mean_js"]))
        mean_overlap_list.append(float(s_res["mean_topk_overlap"]))

    mean_js_arr = np.array(mean_js_list, dtype=float)
    mean_overlap_arr = np.array(mean_overlap_list, dtype=float)

    plt.figure(figsize=(6, 6))
    plt.scatter(mean_js_arr, mean_overlap_arr)

    # 在点附近加 label
    for x, y, label in zip(mean_js_arr, mean_overlap_arr, styles):
        plt.text(x, y, label, fontsize=8, ha="right", va="bottom")

    plt.xlabel("Mean JS divergence across layers")
    plt.ylabel("Mean top-k expert overlap across layers")
    plt.title(
        "Style-level routing shift vs overlap\n"
        f"Model: {metrics.get('model', '')}"
    )
    plt.tight_layout()

    out_path = os.path.join(out_dir, "style_js_vs_overlap_scatter.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Info] Saved style scatter plot to {out_path}")


# ---------- 主入口 ---------- #

def main():
    parser = argparse.ArgumentParser(
        description="Visualize MoE routing metrics from JSON."
    )
    parser.add_argument(
        "--json_path",
        default="Data/Output/moe_routing_style_analysis.json",
        type=str,
        help="Path to the metrics JSON file (from moe_routing_style_analysis).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="figures_moe_routing",
        help="Directory to save generated figures.",
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    print(f"[Info] Loading metrics from {args.json_path}")
    metrics = load_metrics(args.json_path)

    # 依次画不同的图
    plot_per_layer_js_lines(metrics, args.out_dir)
    plot_per_layer_js_heatmap(metrics, args.out_dir)
    plot_per_layer_overlap_heatmap(metrics, args.out_dir)
    plot_style_js_vs_overlap_scatter(metrics, args.out_dir)

    print("[Info] All figures generated.")


if __name__ == "__main__":
    main()
