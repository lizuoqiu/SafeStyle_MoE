import json
import random
from typing import Dict, Tuple, List

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM


# ======== 你需要改的配置 ========
CSV_PATH = "Data/Input/jailbreaks_literary_short_prompt_with_paraphrase.csv"
# 每个 style 最多画多少个 token 的折线
MAX_TOKENS_PER_STYLE_FOR_LINES = 12

# 想要展示的“真实层号”（1-based），比如 2/4/6/8/10/12/14 层
SELECT_LAYERS_DISPLAY = [2, 4, 6, 8, 10, 12, 14]

# 插值后每条曲线的采样点数量（越大越顺）
N_INTERP_POINTS = 200


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


def plot_token_routing_pair(
    model_name: str,
    row_idx: int,
    style_name: str,
    orig_tokens: List[str],
    orig_matrix: np.ndarray,
    style_tokens: List[str],
    style_matrix: np.ndarray,
    save_path: str,
):
    """
    画一张图：左 Original，右 当前 style
      - x 轴：真实层号（2,4,6,8,10,12,14）
      - y 轴：Top-1 Expert ID（在这些层上是精确的整数，中间用插值平滑）
      - 每条折线：一个 token 在这些层上的 expert 轨迹（插值成光滑曲线）
    """
    num_layers, seq_len_orig = orig_matrix.shape
    num_layers2, seq_len_style = style_matrix.shape
    assert num_layers == num_layers2, "orig/style 层数不一致？"

    # ===== 选出在当前模型中有效的层 =====
    # SELECT_LAYERS_DISPLAY 是 1-based（2/4/6/...）
    internal_indices = []
    display_layers = []
    for L in SELECT_LAYERS_DISPLAY:
        idx = L - 1  # 转成 0-based 索引
        if idx < num_layers:
            internal_indices.append(idx)
            display_layers.append(L)

    if not internal_indices:
        # 如果模型层数太少，就退回用所有层
        internal_indices = list(range(num_layers))
        display_layers = [i + 1 for i in internal_indices]

    internal_indices = np.array(internal_indices)        # 0-based，用来从矩阵里取行
    x_sample = np.array(display_layers, dtype=float)     # 1-based，用来当 x 轴（2,4,6,...）
    x_dense = np.linspace(x_sample[0], x_sample[-1], N_INTERP_POINTS)

    # 前 N 个 token
    n_tok_orig = min(seq_len_orig, MAX_TOKENS_PER_STYLE_FOR_LINES)
    n_tok_style = min(seq_len_style, MAX_TOKENS_PER_STYLE_FOR_LINES)

    # 最后一层最后 token 的真实 expert（用于标题）
    last_expert_orig = int(orig_matrix[-1, min(seq_len_orig - 1, MAX_LEN - 1)])
    last_expert_style = int(style_matrix[-1, min(seq_len_style - 1, MAX_LEN - 1)])

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    # ---------- 左图：Original ----------
    ax0 = axes[0]
    for t_idx in range(n_tok_orig):
        # 在选中的层上拿 expert 序列
        y_sample = orig_matrix[internal_indices, t_idx].astype(float)  # [len(display_layers)]
        # 在这些 layer 上做插值
        y_dense = np.interp(x_dense, x_sample, y_sample)

        # token 文本（如果想去掉 Ġ，可以用 tokenizer.convert_tokens_to_string 再处理）
        label = orig_tokens[t_idx]

        # 画平滑曲线
        ax0.plot(x_dense, y_dense, linewidth=2, label=label)
        # 在真实层号上打点（这里的点就是精确的 expert ID）
        ax0.scatter(x_sample, y_sample, s=20)

    ax0.set_title("Original", fontsize=11)
    ax0.set_xlabel("MoE Layer")
    ax0.set_ylabel("Expert ID (Top-1; exact at ticks)")
    ax0.grid(True, linestyle="--", alpha=0.4)
    ax0.legend(fontsize=7, ncol=2, title=f"Tokens (first {n_tok_orig})")

    # ---------- 右图：当前 style ----------
    ax1 = axes[1]
    for t_idx in range(n_tok_style):
        y_sample = style_matrix[internal_indices, t_idx].astype(float)
        y_dense = np.interp(x_dense, x_sample, y_sample)

        label = style_tokens[t_idx]
        ax1.plot(x_dense, y_dense, linewidth=2, label=label)
        ax1.scatter(x_sample, y_sample, s=20)

    ax1.set_title(style_name, fontsize=11)
    ax1.set_xlabel("MoE Layer")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=7, ncol=2, title=f"Tokens (first {n_tok_style})")

    # ---------- 统一 x 轴刻度，显示真实层号 ----------
    ax0.set_xticks(x_sample)
    ax0.set_xticklabels([str(int(x)) for x in x_sample])
    ax1.set_xticks(x_sample)
    ax1.set_xticklabels([str(int(x)) for x in x_sample])

    fig.suptitle(
        f"Model: {model_name}\nRow #{row_idx} | Style: {style_name}  |  "
        f"Last-layer last-token expert: Original={last_expert_orig}, {style_name}={last_expert_style}",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
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
        plot_token_routing_pair(
            TARGET_MODEL,
            rand_idx,
            style_name,
            orig_tokens,
            orig_mat,
            style_tokens,
            style_mat,
            save_path,
        )

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
