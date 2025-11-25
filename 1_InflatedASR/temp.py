import json
import random
from typing import Dict, Tuple, List

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from transformers import AutoTokenizer, AutoModelForCausalLM


# ======== 你需要改的配置 ========
CSV_PATH = "Data/Input/jailbreaks_literary_short_prompt_with_paraphrase.csv"
# 每个 style 最多画多少个 token 的折线
MAX_TOKENS_PER_STYLE_FOR_LINES = 12

# 想要展示的“真实层号”（1-based），比如 2/4/6/8/10/12/14 层
SELECT_LAYERS_DISPLAY = [2, 4, 6, 8, 10, 12, 14]
# SELECT_LAYERS_DISPLAY = list(range(2, 15))
# 插值后每条曲线的采样点数量（越大越顺）
N_INTERP_POINTS = 200
FOCUS_STYLE = "shakespeare"   # 你可以改成 "paraphrase" 等


# key: 图例里显示的风格名字
# value: CSV 里的对应列名
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
# 要分析的模型（你可以改成 6 个里面随便一个）
TARGET_MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128        # 单条 prompt 最多保留多少 token，避免图太宽
RANDOM_SEED = 42     # 用来“随便抽一行”，又保证复现性

# 画图时 x 轴最多显示多少个 token 的标签（避免挤成一坨）
MAX_TOKENS_FOR_XTICK = 25


# ======== 工具函数 ========
def majority_vote_np(arr: np.ndarray) -> int:
    """对一维数组做多数表决，返回出现次数最多的值"""
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])

def aggregate_prompt_route(
    experts_matrix: np.ndarray,
    internal_indices: np.ndarray,
) -> np.ndarray:
    """
    experts_matrix: [num_layers, seq_len]
    internal_indices: 选出来要看的层（0-based）
    返回: [num_selected_layers]，每一层的“主 expert”（对该层所有 token 多数表决）
    """
    routes = []
    for idx in internal_indices:
        layer_experts = experts_matrix[idx]  # [seq_len]
        routes.append(majority_vote_np(layer_experts))
    return np.array(routes, dtype=float)


def get_token_level_experts(
    model,
    tokenizer,
    text: str,
) -> Tuple[List[str], np.ndarray]:
    """
    对一条 text：
      - 返回 token 序列（字符串）
      - 返回 experts_matrix: [num_layers, seq_len]，元素是 Top-1 expert id（int）
    """
    # 编码
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_router_logits=True,  # 关键：要 router_logits
            use_cache=False,
            return_dict=True,
        )

    if not hasattr(outputs, "router_logits") or outputs.router_logits is None:
        # 某些模型可能叫 router_probs，这里做个降级
        if hasattr(outputs, "router_probs") and outputs.router_probs is not None:
            router_logits = tuple(torch.log(p) for p in outputs.router_probs)
        else:
            raise ValueError("Model outputs do not contain router_logits/router_probs")

    else:
        router_logits = outputs.router_logits

    # 转成 token 序列，剥掉 batch 维
    input_ids = inputs["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    seq_len = len(tokens)

    experts_per_layer: List[List[int]] = []

    for layer_id, logits in enumerate(router_logits):
        # logits 形状可能是：
        # [batch, seq_len, n_experts] 或 [seq_len, n_experts] 或 [batch, n_experts] 或 [n_experts]
        logits = logits.detach().cpu()

        if logits.dim() == 3:
            # [batch, seq_len, n_experts]
            logits = logits[0]                      # [seq_len, n_experts]
        elif logits.dim() == 2:
            # [seq_len, n_experts] 或 [batch, n_experts]
            if logits.shape[0] == seq_len:
                pass                                # 就当 [seq_len, n_experts]
            else:
                # 当成 [batch, n_experts]，只有一个“token”
                logits = logits[0].unsqueeze(0)     # [1, n_experts]
        elif logits.dim() == 1:
            # [n_experts]
            logits = logits.unsqueeze(0)            # [1, n_experts]
        else:
            raise ValueError(f"Unexpected router_logits dim: {logits.shape}")

        # 对每个 token 取 Top-1 expert
        token_top1 = torch.argmax(logits, dim=-1)    # [seq_len']，不一定等于原 seq_len

        # 如果长度和原 token 不同，做简单对齐（截断或重复最后一个）
        token_experts = token_top1.tolist()
        if len(token_experts) < seq_len:
            token_experts = token_experts + [token_experts[-1]] * (seq_len - len(token_experts))
        elif len(token_experts) > seq_len:
            token_experts = token_experts[:seq_len]

        experts_per_layer.append(token_experts)

    experts_matrix = np.array(experts_per_layer, dtype=np.int64)  # [num_layers, seq_len]
    return tokens, experts_matrix



def plot_prompt_routing_two_lines(
    model_name: str,
    row_idx: int,
    style_name: str,
    orig_matrix: np.ndarray,
    style_matrix: np.ndarray,
    save_path: str,
):
    """
    只画一张图，两条线：
      - 原句的层级路由 (Original)
      - style 改写后的层级路由 (style_name)

    x 轴：MoE Layer（比如 2~14）
    y 轴：该层上全句 token 的“主 expert ID”（多数表决 Top-1）

    在 2~14 这些整数层上，点是精确的 expert ID；
    中间用三次样条插值变成一条顺滑的曲线。
    """
    num_layers, _ = orig_matrix.shape
    num_layers2, _ = style_matrix.shape
    assert num_layers == num_layers2, "orig/style 层数不一致？"

    # 1) 选出有效层（1-based 的 2/3/.../14 → 0-based 索引）
    internal_indices = []
    display_layers = []
    for L in SELECT_LAYERS_DISPLAY:   # 现在是 [2..14]
        idx = L - 1
        if idx < num_layers:
            internal_indices.append(idx)
            display_layers.append(L)
    if not internal_indices:  # 层数不够就全部用上
        internal_indices = list(range(num_layers))
        display_layers = [i + 1 for i in internal_indices]

    internal_indices = np.array(internal_indices)
    x_sample = np.array(display_layers, dtype=float)                # 真正的 layer 编号（2,3,...）
    # 用更密的 x 做插值（视觉上更顺滑）
    x_dense = np.linspace(x_sample[0], x_sample[-1], N_INTERP_POINTS)

    # 2) 对每层做一次“全句多数表决”，得到每层主 expert
    y_orig_sample = aggregate_prompt_route(orig_matrix, internal_indices)   # [num_layers_sel]
    y_style_sample = aggregate_prompt_route(style_matrix, internal_indices)

    # 3) 三次样条插值，如果失败则退回线性插值
    def smooth_curve(x, y):
        if len(x) >= 4:
            try:
                spline = make_interp_spline(x, y, k=3)
                return spline(x_dense)
            except Exception:
                pass
        # 回退：线性插值
        return np.interp(x_dense, x, y)

    y_orig_dense = smooth_curve(x_sample, y_orig_sample)
    y_style_dense = smooth_curve(x_sample, y_style_sample)

    # 4) 标注：最后一层的主 expert（全句多数表决）
    last_expert_orig = int(y_orig_sample[-1])
    last_expert_style = int(y_style_sample[-1])

    plt.figure(figsize=(10, 5))

    # 两条“漂亮的曲线”
    plt.plot(x_dense, y_orig_dense, linewidth=3, label="Original")
    plt.plot(x_dense, y_style_dense, linewidth=3, linestyle="--", label=style_name)

    # 在真实层号上打点（这里的点就是精确的 expert ID）
    plt.scatter(x_sample, y_orig_sample, s=40)
    plt.scatter(x_sample, y_style_sample, s=40)

    plt.xlabel("MoE Layer")
    plt.ylabel("Expert ID (Top-1 majority per layer)")
    plt.xticks(x_sample, [str(int(x)) for x in x_sample])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    plt.title(
        f"{model_name}\nRow #{row_idx} | Original vs {style_name} | "
        f"Last-layer majority expert: Orig={last_expert_orig}, {style_name}={last_expert_style}",
        fontsize=12
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved figure: {save_path}")

def plot_prompt_routing_all_styles(
    model_name: str,
    row_idx: int,
    routes: Dict[str, np.ndarray],
    display_layers: List[int],
    save_path: str,
):
    """
    一张图画所有 style 的层级路由：

    routes: dict[style_name -> y_sample]
        其中 y_sample 是在 display_layers 上，每层的“主 expert ID”（多数表决之后）
        例如 y_sample.shape = [len(display_layers)]
    display_layers: list[int]
        真实 layer 编号（1-based），比如 [2,3,...,14]
    """
    # x 轴：真实层号
    x_sample = np.array(display_layers, dtype=float)
    # 更密的 x，用来插值成曲线
    x_dense = np.linspace(x_sample[0], x_sample[-1], N_INTERP_POINTS)

    def smooth_curve(x, y):
        # y: np.ndarray，长度和 x 一样
        if len(x) >= 4:
            try:
                spline = make_interp_spline(x, y, k=3)
                return spline(x_dense)
            except Exception:
                pass
        # 回退到线性插值
        return np.interp(x_dense, x, y)

    plt.figure(figsize=(10, 6))

    # 为了让 legend 稍微好看一些，Original 放在最前
    all_styles = list(routes.keys())
    if "Original" in all_styles:
        all_styles.remove("Original")
        all_styles = ["Original"] + all_styles

    for style_name in all_styles:
        y_sample = routes[style_name]          # [len(display_layers)]
        y_sample = np.asarray(y_sample, dtype=float)

        # 平滑曲线
        y_dense = smooth_curve(x_sample, y_sample)

        # 画曲线
        if style_name == "Original":
            # Original 用实线强调一下
            plt.plot(x_dense, y_dense, linewidth=3, label=style_name)
        else:
            plt.plot(x_dense, y_dense, linewidth=2, linestyle="--", label=style_name)

        # 在真实 layer 上打点（精确的 expert ID）
        plt.scatter(x_sample, y_sample, s=25)

    plt.xlabel("MoE Layer")
    plt.ylabel("Expert ID (Top-1 majority per layer)")
    plt.xticks(x_sample, [str(int(x)) for x in x_sample])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=8, ncol=2)

    # 标题：简单说清楚是“一个 prompt 下，所有 style 的 routing”
    plt.title(
        f"{model_name}\nRow #{row_idx} | Routing pathways of Original and all styles",
        fontsize=12
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved figure: {save_path}")



def main():
    print(f"Loading CSV from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # 检查列是否存在
    for style_name, col in STYLE_COLUMNS.items():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' for style '{style_name}' not found in CSV!")

    # 随机抽取一行（固定 seed）
    random.seed(RANDOM_SEED)
    rand_idx = random.choice(df.index.tolist())
    rand_idx = 1
    row = df.loc[rand_idx]
    print(f"Randomly selected row index: {rand_idx}")

    # 加载模型
    print(f"Loading model: {TARGET_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL,
        dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    model.to(DEVICE)
    model.eval()

    # 输出：这一行所有风格的 token-level routing
    all_tokens = {}
    all_mats = {}

    for style_name, col in STYLE_COLUMNS.items():
        text = row[col]
        if pd.isna(text):
            print(f"[WARN] style {style_name} has NaN text in this row, skip")
            continue
        text = str(text)
        print(f"\n=== Style: {style_name} ===")
        print(f"Text: {text[:150]}{'...' if len(text) > 150 else ''}")

        tokens, mat = get_token_level_experts(model, tokenizer, text)
        all_tokens[style_name] = tokens
        all_mats[style_name] = mat

    if "Original" not in all_mats:
        raise ValueError("STYLE_COLUMNS 必须包含 'Original' 这个 key，并且该行不能为 NaN！")

    orig_tokens = all_tokens["Original"]
    orig_mat = all_mats["Original"]

    # 对每个 style（排除 Original），画一张“Original vs style”的图
    for style_name in STYLE_COLUMNS.keys():
        if style_name == "Original":
            continue
        if style_name not in all_mats:
            print(f"[WARN] style {style_name} has no matrix (可能是 NaN 被跳过)，跳过画图")
            continue

        style_tokens = all_tokens[style_name]
        style_mat = all_mats[style_name]

        save_path = f"token_routing_row{rand_idx}_{TARGET_MODEL.replace('/', '_')}_{style_name}.png"
        plot_prompt_routing_two_lines(
            TARGET_MODEL,
            rand_idx,
            style_name,
            orig_mat,
            style_mat,
            save_path,
        )
    # ---------- 计算：这一行 prompt 在每个 style 下的“层级路由” ----------
    # orig_mat = all_mats["Original"]
    # num_layers, _ = orig_mat.shape
    #
    # # 选出有效层（1-based 2..14 → 0-based index）
    # internal_indices = []
    # display_layers = []
    # for L in SELECT_LAYERS_DISPLAY:
    #     idx = L - 1
    #     if idx < num_layers:
    #         internal_indices.append(idx)
    #         display_layers.append(L)
    # if not internal_indices:
    #     internal_indices = list(range(num_layers))
    #     display_layers = [i + 1 for i in internal_indices]
    #
    # internal_indices = np.array(internal_indices)
    #
    # # 对每个 style，算出在这些层上的“主 expert 路径”
    # routes = {}
    # for style_name, mat in all_mats.items():
    #     # mat: [num_layers, seq_len]
    #     y_sample = aggregate_prompt_route(mat, internal_indices)  # [len(display_layers)]
    #     routes[style_name] = y_sample
    #
    # # ---------- 画一张总图：Original + 所有 style ----------
    # save_path = f"prompt_routing_row{rand_idx}_{TARGET_MODEL.replace('/', '_')}_ALL_STYLES.png"
    # plot_prompt_routing_all_styles(
    #     TARGET_MODEL,
    #     rand_idx,
    #     routes,
    #     display_layers,
    #     save_path,
    # )

    # 也顺手把这一行的所有 routing 存成 json，方便以后分析
    out_json = f"token_routing_row{rand_idx}_{TARGET_MODEL.replace('/', '_')}.json"
    serializable = {
        style: {
            "tokens": all_tokens[style],
            "experts": all_mats[style].tolist(),
        }
        for style in all_mats.keys()
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"Saved raw routing data to {out_json}")


if __name__ == "__main__":
    main()
