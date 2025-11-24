import json
import math
from collections import Counter, defaultdict

import torch
import pandas as pd
import matplotlib.pyplot as plt
from dotenv.parser import Original
from transformers import AutoTokenizer, AutoModelForCausalLM


# ====== 基本配置 ======
CSV_PATH = "Data/Input/jailbreaks_literary_short_prompt_with_paraphrase.csv"

STYLE_COLUMNS = {
    "Original":"Original Query",
    "shakespeare":"shakespeare Query" ,
    "biblical":"biblical Query",
    "opera":"opera Query",
    "epic_poetry":"epic_poetry Query",
    "noir": "noir Query",
    "high_fantasy":"high_fantasy Query",
    "cyberpunk": "cyberpunk Query",
    "scientific": "scientific Query",
    "news": "news Query",
    "bedtime" : "bedtime Query",
    "philosophical" : "philosophical Query",
    "mythological" : "mythological Query",
    "paraphrase" : "paraphrase Query" ,
}

# 要跑的所有 MoE 模型
MODELS = [
    "allenai/OLMoE-1B-7B-0924",
    "allenai/OLMoE-1B-7B-0924-Instruct",
    "Qwen/Qwen1.5-MoE-A2.7B",
    "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]

MAX_SAMPLES_PER_STYLE = 1200   # 每种 style 最多抽多少条 query 来算主流 routing
MAX_LEN = 512                 # 编码时的最大长度
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def majority_vote(xs):
    """
    xs 可以是 list/tuple/tensor/标量，返回出现次数最多的元素
    """
    if isinstance(xs, torch.Tensor):
        xs = xs.tolist()

    # 标量情况：变成单元素列表
    if not isinstance(xs, (list, tuple)):
        xs = [xs]

    c = Counter(xs)
    return c.most_common(1)[0][0]

# ====== 单个 query：从 router_logits 算出每一层的 Top-1 expert（对 token 多数表决） ======
def query_route_from_router_logits(router_logits):
    """
    router_logits: tuple[num_layers] of tensor[ ... , n_experts ]
    不管前面有几维（batch / seq_len 等），全部 flatten 到 token 维度，
    只保留最后一维作为 expert 维度。
    返回: {layer_id: expert_id}
    """
    layer_to_expert = {}

    for layer_id, logits in enumerate(router_logits):
        # logits 可能是 [batch, seq_len, n_experts] / [batch, n_experts] / [n_experts]
        logits = logits.detach().cpu()

        n_experts = logits.shape[-1]
        logits_flat = logits.reshape(-1, n_experts)  # [num_tokens, n_experts]

        # 对每个“token”取 Top-1 expert
        token_top1 = torch.argmax(logits_flat, dim=-1)  # [num_tokens]

        # 对所有 token 做多数表决，得到这一层的主 expert
        expert_id = majority_vote(token_top1)
        layer_to_expert[layer_id] = int(expert_id)

    return layer_to_expert



# ====== 一个 style：对很多 query 聚合成“style 级别主路由” ======
def aggregate_style_routes(model, tokenizer, prompts):
    """
    对某个 style 的一堆 prompts：
      - 每个 prompt forward 一次拿 router_logits
      - 先算 query 级别的 {layer -> expert}（token 上多数表决）
      - 再对所有 query 的 expert 做多数表决，得到 style 级别 {layer -> expert}
    """
    per_layer_experts = defaultdict(list)  # layer_id -> [expert_ids from different queries]

    for idx, prompt in enumerate(prompts):
        if isinstance(prompt, float) and pd.isna(prompt):
            continue
        text = str(prompt)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN
        ).to(DEVICE)

        with torch.no_grad():
            # 关键：让模型返回 router_logits（不同 MoE 模型都支持这个参数）
            outputs = model(
                **inputs,
                output_router_logits=True,
                return_dict=True,
                use_cache=False,
            )

        router_logits = outputs.router_logits  # tuple[layer] of [1, seq_len, n_experts]
        query_route = query_route_from_router_logits(router_logits)

        for layer_id, expert_id in query_route.items():
            per_layer_experts[layer_id].append(expert_id)

        if (idx + 1) % 10 == 0:
            print(f"    processed {idx + 1}/{len(prompts)} prompts")

    # 对每一层再做一次多数表决，得到 style 的主路由
    style_route = {
        layer_id: majority_vote(expert_ids)
        for layer_id, expert_ids in per_layer_experts.items()
        if len(expert_ids) > 0
    }
    return style_route


# ====== 单个模型：计算所有 style 的 routing ======
def compute_routes_for_model(model_name, df):
    print(f"\n=== Loading model: {model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    model.eval()

    # 某些实现需要在 config 里显式打开
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = True

    style_routes = {}

    for style_name, col in STYLE_COLUMNS.items():
        if col not in df.columns:
            print(f"[WARN] column {col} not in CSV, skip style {style_name}")
            continue

        prompts = df[col].dropna().tolist()[:MAX_SAMPLES_PER_STYLE]
        print(f"  Style: {style_name}, #prompts={len(prompts)}")

        style_route = aggregate_style_routes(model, tokenizer, prompts)
        style_routes[style_name] = style_route

    # 单模型保存一个 json
    out_json = f"routing_{model_name.replace('/', '_')}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(style_routes, f, ensure_ascii=False, indent=2)
    print(f"  Saved routing data for {model_name} to {out_json}")

    # 释放显存
    del model
    del tokenizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return style_routes


# ====== 画“大图”：多个模型 × 多个 style 的 routing ======
def plot_all_models_style_routes(all_routes, save_path="routing_all_models.png"):
    """
    all_routes: dict[model_name -> dict[style_name -> {layer_id: expert_id}]]
    """
    n_models = len(all_routes)
    if n_models == 0:
        print("No routes to plot.")
        return

    n_cols = 2
    n_rows = math.ceil(n_models / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 3 * n_rows),
        sharex=False, sharey=False
    )
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 固定 style 的颜色/label 顺序
    style_names = list(next(iter(all_routes.values())).keys())

    for i, (model_name, style_routes) in enumerate(all_routes.items()):
        ax = axes[i]

        for style in style_names:
            if style not in style_routes:
                continue
            layer_map = style_routes[style]
            layers = sorted(layer_map.keys())
            experts = [layer_map[l] for l in layers]
            ax.plot(layers, experts, marker="o", label=style)

        ax.set_title(model_name, fontsize=9)
        ax.set_xlabel("MoE Layer")
        ax.set_ylabel("Top-1 Expert ID\n(majority over queries)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=7)

    # 多余的 subplot 去掉坐标
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "MoE Routing (Top-1, majority over tokens & queries)\n"
        "Comparison of styles across models",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save_path, dpi=300)
    print(f"Saved big figure to {save_path}")


def main():
    df = pd.read_csv(CSV_PATH)

    all_routes = {}

    for model_name in MODELS:
        style_routes = compute_routes_for_model(model_name, df)
        all_routes[model_name] = style_routes

    # 保存一个总的 json，里面包含所有模型 & style 的 routing
    with open("routing_all_models.json", "w", encoding="utf-8") as f:
        json.dump(all_routes, f, ensure_ascii=False, indent=2)
    print("Saved aggregated routing data to routing_all_models.json")

    # 画一张“大图”
    plot_all_models_style_routes(all_routes, save_path="routing_all_models.png")


if __name__ == "__main__":
    main()
