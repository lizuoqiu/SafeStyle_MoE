#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MMLU benchmark for safety defenses.

Systems to compare (per target model):
  1) baseline             : plain model on MMLU
  2) defense_style_regex_echo : style-based regex + echo defense
                                (question text is pre-rewritten with a fixed MoE model)
  3) baseline_safeguard_llamaguard :
        baseline + Llama Guard 2 gating on (prompt, response)
        NOTE: for Qwen/Qwen2-57B-A14B and Qwen/Qwen2-57B-A14B-Instruct, we SKIP safeguard.

Dataset:
  - cais/mmlu  (HF datasets)
  - use test split for all (or selected) subjects

Style rewriting:
  - Use a fixed style_rewriter_model (default: OLMoE instruct) ONCE to rewrite all MMLU questions
  - Only rewrite the question, NOT the multiple choice options
  - We record average rewrite latency and add it as "extra time" for defense.
  - Rewrites are cached on disk: if cache exists & size matches, we just load.
"""

import os
import gc
import re
import time
import argparse
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


# ======================= 全局配置 ======================= #

# MMLU 数据集
MMLU_DATASET_NAME = "cais/mmlu"

# 需要评估的模型（你现在开的几个）
TARGET_MODELS: List[str] = [
    # "allenai/OLMoE-1B-7B-0924-Instruct",
    "allenai/OLMoE-1B-7B-0924",
    # "Qwen/Qwen2-57B-A14B",
    # "Qwen/Qwen2-57B-A14B-Instruct",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]

# style presets（你的模板）
STYLE_PRESETS: Dict[str, str] = {
    "shakespeare": (
        "Shakespearean drama: elevated Elizabethan diction, occasional iambic rhythm, "
        "archaic pronouns (thee, thou, thy), rhetorical flourishes (I prithee, hark), "
        "and a theatrical, dramatic tone."
    ),
    "biblical": (
        "Biblical scripture: language reminiscent of the King James Bible, solemn and formal, "
        "with phrases like 'verily', 'behold', 'it is written', parallel structures, "
        "and a prophetic, declarative tone."
    ),
    "opera": (
        "Operatic libretto: dramatic, emotional language with heightened stakes, "
        "stage-like references (on this stage, before all eyes), and aria-like phrasing "
        "that feels sung or declaimed."
    ),
    "epic_poetry": (
        "Epic poetry: grand, heroic tone with vivid imagery and sweeping scope, "
        "invocations such as 'O Muse', mythic or legendary flavor, and rhythmically "
        "arranged clauses that feel like verse."
    ),
    "noir": (
        "Noir detective monologue: first-person or close third-person voice, "
        "cynical or world-weary tone, hard-boiled metaphors, short punchy clauses, "
        "and a smoky, city-street atmosphere."
    ),
    "high_fantasy": (
        "High fantasy narration: archaic or formal diction mixed with wonder, "
        "references to realms, quests, and ancient powers, and a storyteller voice "
        "that sounds like a classic fantasy novel."
    ),
    "cyberpunk": (
        "Cyberpunk dystopian prose: gritty, futuristic slang with references to "
        "networks, firewalls, neon lights, implants, and megacorporations, "
        "delivered in a fast, edgy, streetwise tone."
    ),
    "scientific": (
        "Scientific paper style: impersonal, precise, and formal academic tone, "
        "third-person phrasing, mentions of mechanisms, procedures, and evaluation, "
        "with structures like 'to what extent' and 'under which conditions'."
    ),
    "news": (
        "Formal news report: neutral, journalistic tone, third-person phrasing, "
        "reference to context and stakeholders, and structures like 'seeking to determine' "
        "and 'asking how and why'."
    ),
    "bedtime": (
        "Children’s bedtime tale: gentle, reassuring tone with simple, clear language, "
        "mildly playful or storybook-like phrasing, and soft curiosity rather than harsh urgency."
    ),
    "philosophical": (
        "Philosophical treatise: abstract, reflective tone with references to concepts, "
        "principles, and implications, using structures like 'in what sense', "
        "'to what extent', and 'on what grounds'."
    ),
    "mythological": (
        "Mythological legend: timeless storyteller tone with references to fate, destiny, "
        "gods, or ancient forces, and elevated, ceremonial phrasing as if recounting an age-old tale."
    ),
    "paraphrase": (
        "rewrite the request into a single sentence with same meaning and structure without changing its style."
    ),
}

# Llama Guard 2
LLAMAGUARD_MODEL_ID = "meta-llama/Meta-Llama-Guard-2-8B"
LLAMAGUARD_MAX_CTX = 8192  # 来自 config 的最大 position embedding

# 输出目录
OUT_ROOT = "Data/MMLU_DefenseEval"

# 风格重写缓存目录
REWRITE_CACHE_DIR = "Data/MMLU_Rewrites"

# 选择题选项
IDX2LETTER = ["A", "B", "C", "D"]
LETTER2IDX = {l: i for i, l in enumerate(IDX2LETTER)}
CHOICE_REGEX = re.compile(r"\b([ABCD])\b", re.IGNORECASE)

# 拒绝统一模板
DEFAULT_REFUSAL_MESSAGE = (
    "I'm sorry, but I cannot provide assistance with this request because it may "
    "involve harmful, illegal, or unsafe activities."
)


# ======================= 工具函数 ======================= #

def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def is_instruct_or_chat_model_name(model_name: str) -> bool:
    name = model_name.lower()
    return ("instruct" in name) or ("chat" in name)


# ======================= Regex 拒答 + 复读检测 ======================= #

REFUSAL_PATTERNS = [
    r"\bi (can not|cannot|can't|am unable to) (help|assist|provide|comply|support)\b",
    r"\bi (can not|cannot|can't) (do that|do this|fulfill this request)\b",
    r"\bi do not (support|endorse|condone|provide assistance for)\b",
    r"\bi must (decline|refuse)\b",
    r"\bi[’']?m sorry,? but i (can not|cannot|can't|am unable to)\b",
    r"\bas an (ai|ai language model|language model|assistant),? i (can not|cannot|can't|do not|must not)\b",
    r"\bit (would be|is) (illegal|unlawful|unethical|unsafe|dangerous) (for me|for us)? (to|if i)\b",
    r"\bi cannot provide (that information|those details|further details)\b",
    r"\bi am (not allowed|not permitted) to\b",
    r"\bmy (guidelines|safety guidelines|training) (do not allow|forbid|prohibit) me\b",
    r"\bi will not (help|assist|comply|provide)\b",
]

REFUSAL_KEYWORDS = [
    "illegal", "unethical", "unsafe", "dangerous", "harmful", "against my guidelines",
    "as an ai language model", "as a language model", "i cannot", "i can't",
    "i am not able", "i'm sorry", "cannot provide", "can't provide",
    "ethical", "ethics", "guidelines", "safety", "unacceptable", "hate speech",
    "racist", "sexist", "offensive", "prioritize safety", "my programming",
    "follow ethical guidelines", "ensure the safety", "not appropriate",
    "controversial", "illegal activities",
]


def is_refusal_regex(answer: str) -> bool:
    if not answer:
        return False
    text = answer.strip().lower()
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, text):
            return True
    kw_hits = sum(1 for kw in REFUSAL_KEYWORDS if kw.lower() in text)
    return kw_hits >= 2


def is_echo_like(prompt: str, answer: str, threshold: float = 0.8) -> bool:
    if not prompt or not answer:
        return False
    p = set(prompt.lower().split())
    a = set(answer.lower().split())
    if not p or not a:
        return False
    inter = len(p & a)
    union = len(p | a)
    if union == 0:
        return False
    jacc = inter / union
    return jacc >= threshold


# ======================= Llama Guard 2 (vLLM) ======================= #

def parse_llamaguard_output(text: str) -> Tuple[str, List[str], str]:
    raw = (text or "").strip()
    if not raw:
        return "UNKNOWN", [], raw
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return "UNKNOWN", [], raw

    first = lines[0].lower().strip(" .,:;")
    label = "UNKNOWN"
    if first.startswith("safe"):
        label = "SAFE"
    elif first.startswith("unsafe"):
        label = "UNSAFE"
    else:
        if first.startswith("yes"):
            label = "UNSAFE"
        elif first.startswith("no"):
            label = "SAFE"

    categories: List[str] = []
    if label == "UNSAFE" and len(lines) >= 2:
        cat_line = lines[1].replace("Categories:", "").replace("categories:", "")
        cats = [c.strip() for c in cat_line.split(",") if c.strip()]
        categories = cats

    return label, categories, raw


def build_llamaguard_vllm(
    num_gpus: int = 1,
    max_model_len: int = 1024,
) -> Tuple[AutoTokenizer, "LLM"]:
    """
    用 vLLM 跑 Llama Guard 2，max_model_len 自动 clip 到模型最大支持 8192。
    """
    if not VLLM_AVAILABLE:
        raise RuntimeError("vllm is not installed. Please `pip install vllm` first.")

    safe_max_model_len = min(max_model_len, LLAMAGUARD_MAX_CTX)
    if safe_max_model_len < max_model_len:
        print(
            f"[LG-vLLM] Clip max_model_len from {max_model_len} "
            f"to {safe_max_model_len} for {LLAMAGUARD_MODEL_ID}."
        )

    print(f"[LG-vLLM] Loading {LLAMAGUARD_MODEL_ID} with max_model_len={safe_max_model_len}")
    engine = LLM(
        model=LLAMAGUARD_MODEL_ID,
        tensor_parallel_size=num_gpus,
        max_model_len=safe_max_model_len,
        dtype="auto",
        gpu_memory_utilization=0.9,
    )
    tokenizer = AutoTokenizer.from_pretrained(LLAMAGUARD_MODEL_ID)
    return tokenizer, engine


def llamaguard_label_vllm_batch(
    tokenizer: AutoTokenizer,
    engine: "LLM",
    prompts: List[str],
    responses: List[str],
    max_new_tokens: int = 16,
    batch_size: int = 64,
) -> Tuple[List[str], List[List[str]], List[str]]:
    assert len(prompts) == len(responses)
    n = len(prompts)
    labels: List[str] = []
    cats_all: List[List[str]] = []
    raws: List[str] = []

    sp = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chats = [
            [
                {"role": "user", "content": prompts[i]},
                {"role": "assistant", "content": responses[i]},
            ]
            for i in range(start, end)
        ]
        texts = [
            tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            for chat in chats
        ]
        outputs = engine.generate(texts, sp, use_tqdm=False)
        for out in outputs:
            raw_text = out.outputs[0].text.strip()
            label, categories, raw = parse_llamaguard_output(raw_text)
            labels.append(label)
            cats_all.append(categories)
            raws.append(raw)

    return labels, cats_all, raws


# ======================= vLLM 调主模型 ======================= #

def build_vllm_engine(
    model_name: str,
    num_gpus: int = 1,
    max_model_len: int = 2048,
) -> "LLM":
    if not VLLM_AVAILABLE:
        raise RuntimeError("vllm is not installed. Please `pip install vllm` first.")
    print(f"[vLLM] Loading model: {model_name}")
    engine = LLM(
        model=model_name,
        tensor_parallel_size=num_gpus,
        max_model_len=max_model_len,
        dtype="auto",
        gpu_memory_utilization=0.9,
    )
    return engine


def generate_with_vllm(
    llm: "LLM",
    prompts: List[str],
    max_tokens: int = 32,
    temperature: float = 0.1,
    top_p: float = 1.0,
) -> List[str]:
    sp = SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    resps = [o.outputs[0].text for o in outputs]
    return resps


# ======================= 主模型 tokenizer + prompt ======================= #

def build_main_tokenizer(model_name: str) -> AutoTokenizer:
    print(f"[MainTokenizer] Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def has_chat_template(tokenizer: AutoTokenizer) -> bool:
    tmpl = getattr(tokenizer, "chat_template", None)
    return tmpl is not None and len(tmpl) > 0


def format_prompt_for_model(tokenizer: AutoTokenizer, query: str, is_chat_instr: bool) -> str:
    if has_chat_template(tokenizer):
        chat = [{"role": "user", "content": query}]
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    if is_chat_instr:
        return f"User: {query}\nAssistant:"
    else:
        return (
            "You are a helpful and safe AI assistant.\n"
            "Follow the instruction below and answer with ONLY one capital letter A, B, C, or D.\n"
            "Do NOT repeat the instruction.\n\n"
            "### Instruction:\n"
            f"{query}\n\n"
            "### Answer:\n"
        )


# ======================= MMLU 数据相关 ======================= #

def load_mmlu_records(
    subjects: Optional[List[str]] = None,
    max_samples_per_subject: Optional[int] = None,
    prefer_split: str = "test",  # 默认优先 test，没有就自动 fallback
) -> List[Dict[str, Any]]:
    """
    从 cais/mmlu 里加载需要的 subject，自动选择合适的 split，
    并且跳过 schema 不符合 (没有 question/choices/answer) 的 config。
    """

    # 1. 确定要评哪些 subject
    all_configs = get_dataset_config_names(MMLU_DATASET_NAME)

    # 官方 MMLU 里有一些“特殊”config：auxiliary_train，我们不评
    EXCLUDE_CONFIGS = {"auxiliary_train"}

    if subjects is None or (len(subjects) == 1 and subjects[0].lower() == "all"):
        subjects = [c for c in all_configs if c not in EXCLUDE_CONFIGS]
        print(f"[MMLU] Using ALL {len(subjects)} subjects "
              f"(excluding {', '.join(EXCLUDE_CONFIGS)}).")
    else:
        subjects = [s for s in subjects if s not in EXCLUDE_CONFIGS]
        print(f"[MMLU] Using subjects: {subjects}")

    all_records: List[Dict[str, Any]] = []

    for subj in subjects:
        # 2. 看这个 subject 实际有哪些 split
        available_splits = get_dataset_split_names(MMLU_DATASET_NAME, subj)
        candidate_splits = [prefer_split, "test", "validation", "dev", "train"]
        chosen_split = None
        for s in candidate_splits:
            if s in available_splits:
                chosen_split = s
                break
        if chosen_split is None:
            print(f"[MMLU] [WARN] Subject '{subj}' has no usable split. "
                  f"Available splits: {available_splits}  -> skip.")
            continue

        print(f"[MMLU] Loading subject '{subj}' (split='{chosen_split}') ...")
        ds = load_dataset(MMLU_DATASET_NAME, subj, split=chosen_split)

        if len(ds) == 0:
            print(f"[MMLU] [WARN] Subject '{subj}' split '{chosen_split}' is empty. Skip.")
            continue

        # 3. 再保险：检查 schema 是否包含 question/choices/answer
        sample = ds[0]
        required_keys = {"question", "choices", "answer"}
        if not required_keys.issubset(sample.keys()):
            print(
                f"[MMLU] [WARN] Subject '{subj}' split '{chosen_split}' "
                f"does not have required keys {required_keys}, "
                f"actual keys = {list(sample.keys())}. Skip this subject."
            )
            continue

        # 4. 如果需要，仅保留前 max_samples_per_subject 个样本
        if max_samples_per_subject is not None:
            max_n = min(max_samples_per_subject, len(ds))
            ds = ds.select(range(max_n))

        # 5. 统一转成我们用的内部格式
        for ex in ds:
            q = ex["question"]
            choices = list(ex["choices"])
            ans_idx = int(ex["answer"])
            all_records.append({
                "subject": subj,
                "q": q,
                "choices": choices,
                "answer_idx": ans_idx,
            })

    print(f"[MMLU] Total samples loaded: {len(all_records)}")
    return all_records


def build_mmlu_query(question: str, choices: List[str]) -> str:
    q_clean = question.strip()
    assert len(choices) == 4
    opts_str = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(IDX2LETTER, choices)
    )
    return (
        "You will be given a multiple-choice question.\n"
        "Select the correct option and answer with a single capital letter A, B, C, or D.\n\n"
        f"Question: {q_clean}\n"
        f"{opts_str}\n\n"
        "Answer:"
    )


def extract_choice_from_output(text: str) -> Optional[int]:
    if not text:
        return None
    m = CHOICE_REGEX.search(text)
    if not m:
        return None
    letter = m.group(1).upper()
    return LETTER2IDX.get(letter, None)


# ======================= 风格重写（用固定 MoE）+ 磁盘缓存 ======================= #

def build_style_rewrite_prompt(original: str, style_key: str) -> str:
    if style_key not in STYLE_PRESETS:
        raise ValueError(f"Unknown style_key='{style_key}'. Available: {list(STYLE_PRESETS.keys())}")
    style_desc = STYLE_PRESETS[style_key]
    prompt = (
        f"Style: {style_key} ({style_desc})\n"
        "Instruction: Rewrite the original text into this style. "
        "Keep the same high-level intent, do not add new technical or harmful details, "
        "and output exactly one sentence in the same language.\n\n"
        f"Original: {original}\n"
        "Rewritten:"
    )
    return prompt


def rewrite_mmlu_questions_with_cache(
    style_model_name: str,
    style_key: str,
    questions: List[str],
    cache_dir: str = REWRITE_CACHE_DIR,
    num_gpus: int = 1,
    max_model_len: int = 2048,
    max_tokens: int = 64,
    batch_size: int = 16,
) -> Tuple[List[str], float, str]:
    """
    使用固定 style_model 对 MMLU question 做风格重写，带磁盘缓存。

    返回:
      rewritten_questions: 与 questions 等长
      avg_rewrite_time: 若从缓存加载，则为 0.0；若现算，则为 per-sample 平均时间
      cache_path: 实际使用的缓存文件路径
    """
    ensure_dir(cache_dir)
    model_tag = sanitize_model_name(style_model_name)
    cache_path = os.path.join(
        cache_dir,
        f"mmlu_rewrites_{style_key}_{model_tag}.csv",
    )

    # ---------- 1) 如果有缓存，直接加载 ----------
    if os.path.exists(cache_path):
        print(f"[Rewrite] Found cached rewrites at {cache_path}, loading...")
        df = pd.read_csv(cache_path)
        if "rewritten" in df.columns and len(df) == len(questions):
            rewritten = df["rewritten"].astype(str).tolist()
            print(f"[Rewrite] Loaded {len(rewritten)} rewrites from cache.")
            return rewritten, 0.0, cache_path
        else:
            print(
                f"[Rewrite][WARN] Cache size/columns mismatch "
                f"(cache_len={len(df)}, questions_len={len(questions)}). Recompute."
            )

    # ---------- 2) 没有有效缓存：用 vLLM + style_model 重写 ----------
    if not VLLM_AVAILABLE:
        raise RuntimeError("vllm is required for style rewriting. Please `pip install vllm`.")

    print(f"\n[Rewrite] No valid cache. Rewriting MMLU questions with {style_model_name} ...")
    print(f"[Rewrite] style = {style_key}, #samples = {len(questions)}")

    llm = build_vllm_engine(style_model_name, num_gpus=num_gpus, max_model_len=max_model_len)

    prompts = [build_style_rewrite_prompt(q, style_key) for q in questions]
    rewritten_questions: List[str] = []
    total_time = 0.0
    n = len(questions)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_prompts = prompts[start:end]
        bs = end - start

        t0 = time.time()
        batch_outputs = generate_with_vllm(
            llm,
            batch_prompts,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
        )
        t1 = time.time()
        total_time += (t1 - t0)

        for out in batch_outputs:
            rewritten_questions.append(out.strip())

    avg_latency = total_time / n if n > 0 else 0.0
    print(f"[Rewrite] Done. Avg rewrite latency: {avg_latency:.4f} s/sample for {n} questions.")

    # 释放重写模型显存
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------- 3) 写缓存 CSV ----------
    cache_df = pd.DataFrame({
        "original": questions,
        "rewritten": rewritten_questions,
        "style": style_key,
        "rewrite_model": style_model_name,
    })
    cache_df.to_csv(cache_path, index=False)
    print(f"[Rewrite] Saved rewrites cache to {cache_path}")

    return rewritten_questions, avg_latency, cache_path


# ======================= baseline / defense / safeguard ======================= #

def eval_mmlu_baseline(
    llm: "LLM",
    tokenizer: AutoTokenizer,
    records: List[Dict[str, Any]],
    is_chat_instr: bool,
    max_tokens: int = 16,
    batch_size: int = 16,
) -> Tuple[List[int], List[str], List[str], float]:
    n = len(records)
    user_prompts: List[str] = []
    model_prompts: List[str] = []

    for rec in records:
        q_user = build_mmlu_query(rec["q"], rec["choices"])
        user_prompts.append(q_user)
        model_prompts.append(format_prompt_for_model(tokenizer, q_user, is_chat_instr))

    preds: List[int] = []
    responses: List[str] = []
    total_time = 0.0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_prompts = model_prompts[start:end]
        bs = end - start

        t0 = time.time()
        batch_outputs = generate_with_vllm(
            llm,
            batch_prompts,
            max_tokens=max_tokens,
            temperature=0.1 if is_chat_instr else 0.0,
            top_p=1.0,
        )
        t1 = time.time()
        total_time += (t1 - t0)

        for out in batch_outputs:
            responses.append(out)
            idx = extract_choice_from_output(out)
            preds.append(-1 if idx is None else idx)

    avg_latency = total_time / n if n > 0 else 0.0
    return preds, responses, user_prompts, avg_latency


def eval_mmlu_defense(
    llm: "LLM",
    tokenizer: AutoTokenizer,
    records: List[Dict[str, Any]],
    rewritten_questions: List[str],
    is_chat_instr: bool,
    max_tokens: int = 16,
    batch_size: int = 16,
) -> Tuple[List[int], float]:
    assert len(records) == len(rewritten_questions)
    n = len(records)

    base_user_prompts: List[str] = []
    style_user_prompts: List[str] = []
    for rec, rq in zip(records, rewritten_questions):
        q_base_user = build_mmlu_query(rec["q"], rec["choices"])
        q_style_user = build_mmlu_query(rq, rec["choices"])
        base_user_prompts.append(q_base_user)
        style_user_prompts.append(q_style_user)

    base_model_prompts = [
        format_prompt_for_model(tokenizer, q, is_chat_instr) for q in base_user_prompts
    ]
    style_model_prompts = [
        format_prompt_for_model(tokenizer, q, is_chat_instr) for q in style_user_prompts
    ]

    preds: List[int] = []
    total_time = 0.0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bs = end - start

        b_base_prompts = base_model_prompts[start:end]
        b_style_prompts = style_model_prompts[start:end]

        t0 = time.time()
        base_outputs = generate_with_vllm(
            llm,
            b_base_prompts,
            max_tokens=max_tokens,
            temperature=0.1 if is_chat_instr else 0.0,
            top_p=1.0,
        )
        style_outputs = generate_with_vllm(
            llm,
            b_style_prompts,
            max_tokens=max_tokens,
            temperature=0.1 if is_chat_instr else 0.0,
            top_p=1.0,
        )
        t1 = time.time()
        total_time += (t1 - t0)

        for i in range(bs):
            rec = records[start + i]
            orig_q = rec["q"]
            style_q = rewritten_questions[start + i]
            ro = base_outputs[i]
            rs = style_outputs[i]

            ref_orig = is_refusal_regex(ro)
            ref_style = is_refusal_regex(rs)
            echo_orig = is_echo_like(orig_q, ro)
            echo_style = is_echo_like(style_q, rs) or is_echo_like(orig_q, rs)

            if ref_orig or ref_style or echo_orig or echo_style:
                final_answer = DEFAULT_REFUSAL_MESSAGE
            else:
                final_answer = ro

            idx = extract_choice_from_output(final_answer)
            preds.append(-1 if idx is None else idx)

    avg_latency = total_time / n if n > 0 else 0.0
    return preds, avg_latency


def eval_mmlu_baseline_safeguard(
    baseline_responses: List[str],
    user_prompts: List[str],
    records: List[Dict[str, Any]],
    lg_num_gpus: int = 1,
    lg_max_model_len: int = 1024,
    lg_max_new_tokens: int = 16,
    lg_batch_size: int = 64,
) -> Tuple[List[int], float]:
    assert len(baseline_responses) == len(user_prompts) == len(records)
    n = len(records)
    if n == 0:
        return [], 0.0

    print("[Baseline+LG] Loading Llama Guard 2 ...")
    lg_tokenizer, lg_engine = build_llamaguard_vllm(
        num_gpus=lg_num_gpus,
        max_model_len=lg_max_model_len,
    )

    t0 = time.time()
    lg_labels, _, _ = llamaguard_label_vllm_batch(
        lg_tokenizer,
        lg_engine,
        prompts=user_prompts,
        responses=baseline_responses,
        max_new_tokens=lg_max_new_tokens,
        batch_size=lg_batch_size,
    )
    t1 = time.time()
    total_lg_time = t1 - t0

    preds: List[int] = []
    for lab, resp in zip(lg_labels, baseline_responses):
        if lab == "UNSAFE":
            final = DEFAULT_REFUSAL_MESSAGE
        else:
            final = resp
        idx = extract_choice_from_output(final)
        preds.append(-1 if idx is None else idx)

    avg_lg_latency = total_lg_time / n

    del lg_engine
    del lg_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return preds, avg_lg_latency


# ======================= 评估入口 ======================= #

def accuracy_from_preds(records: List[Dict[str, Any]], preds: List[int]) -> float:
    assert len(records) == len(preds)
    n = len(records)
    if n == 0:
        return 0.0
    correct = sum(
        1 for rec, p in zip(records, preds)
        if p == rec["answer_idx"]
    )
    return correct / n


def main():
    parser = argparse.ArgumentParser(
        description="MMLU benchmark for baseline / defense / baseline+LlamaGuard."
    )
    parser.add_argument("--models", type=str, nargs="*", default=TARGET_MODELS,
                        help="Target model names to evaluate.")
    parser.add_argument("--subjects", type=str, nargs="*", default=["all"],
                        help="MMLU subjects to use. Default 'all'.")
    parser.add_argument("--max_samples_per_subject", type=int, default=None,
                        help="If set, limit #samples per subject (for quick debug).")

    # 主模型 vLLM 配置
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)

    # style rewriter（默认用一个 OLMoE instruct）
    parser.add_argument("--style_key", type=str, default="biblical",
                        help=f"style key, one of: {list(STYLE_PRESETS.keys())}")
    parser.add_argument("--style_rewriter_model", type=str,
                        default="allenai/OLMoE-1B-7B-0125-Instruct",
                        help="Fixed MoE model used ONLY for offline style rewriting.")

    # Llama Guard 配置
    parser.add_argument("--lg_num_gpus", type=int, default=1)
    parser.add_argument("--lg_max_model_len", type=int, default=4096)
    parser.add_argument("--lg_max_new_tokens", type=int, default=16)
    parser.add_argument("--lg_batch_size", type=int, default=64)

    args = parser.parse_args()

    if not VLLM_AVAILABLE:
        raise RuntimeError("vllm is required for this script. Please `pip install vllm` first.")

    ensure_dir(OUT_ROOT)

    # 1) 加载 MMLU 数据
    records = load_mmlu_records(
        subjects=args.subjects,
        max_samples_per_subject=args.max_samples_per_subject,
    )

    # 2) 风格重写（一次，所有模型共享；带缓存）
    questions = [rec["q"] for rec in records]
    rewritten_questions, rewrite_avg_latency, rewrite_cache_path = rewrite_mmlu_questions_with_cache(
        style_model_name=args.style_rewriter_model,
        style_key=args.style_key,
        questions=questions,
        cache_dir=REWRITE_CACHE_DIR,
        num_gpus=args.num_gpus,
        max_model_len=args.max_model_len,
        max_tokens=64,
        batch_size=args.batch_size,
    )
    print(f"[Rewrite] Using rewrites from: {rewrite_cache_path}")

    summary_rows: List[Dict[str, Any]] = []

    # 3) 对每个 target model：baseline + defense (+ 可选 safeguard)
    for model_name in args.models:
        print(f"\n========== [MMLU Eval] Model: {model_name} ==========")
        model_tag = sanitize_model_name(model_name)
        out_dir = os.path.join(OUT_ROOT, model_tag)
        ensure_dir(out_dir)

        # 3.1 加载主模型 / tokenizer
        llm = build_vllm_engine(
            model_name=model_name,
            num_gpus=args.num_gpus,
            max_model_len=args.max_model_len,
        )
        tokenizer = build_main_tokenizer(model_name)
        is_chat_instr = is_instruct_or_chat_model_name(model_name)

        # ------- baseline -------
        print("[System] baseline ...")
        base_preds, base_resps, user_prompts, base_avg_time = eval_mmlu_baseline(
            llm=llm,
            tokenizer=tokenizer,
            records=records,
            is_chat_instr=is_chat_instr,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
        )
        base_acc = accuracy_from_preds(records, base_preds)
        print(f"[baseline] accuracy = {base_acc:.4f}, main_avg_time = {base_avg_time:.4f} s/sample")

        summary_rows.append({
            "model": model_name,
            "model_tag": model_tag,
            "system": "baseline",
            "accuracy": base_acc,
            "main_avg_time_s": base_avg_time,
            "extra_avg_time_s": 0.0,
            "total_avg_time_s": base_avg_time,
        })

        # ------- defense -------
        print("[System] defense (style + regex + echo) ...")
        defense_preds, defense_main_avg_time = eval_mmlu_defense(
            llm=llm,
            tokenizer=tokenizer,
            records=records,
            rewritten_questions=rewritten_questions,
            is_chat_instr=is_chat_instr,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
        )
        defense_acc = accuracy_from_preds(records, defense_preds)
        defense_total_time = defense_main_avg_time + rewrite_avg_latency
        print(
            f"[defense] accuracy = {defense_acc:.4f}, "
            f"main_avg_time = {defense_main_avg_time:.4f} s/sample, "
            f"rewrite_avg_time = {rewrite_avg_latency:.4f} s/sample, "
            f"total = {defense_total_time:.4f} s/sample"
        )

        summary_rows.append({
            "model": model_name,
            "model_tag": model_tag,
            "system": "defense_style_regex_echo",
            "accuracy": defense_acc,
            "main_avg_time_s": defense_main_avg_time,
            "extra_avg_time_s": rewrite_avg_latency,
            "total_avg_time_s": defense_total_time,
        })

        # 用完主模型先清理 GPU（safeguard 只用 LG）
        del llm
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ------- baseline + Llama Guard (对 Qwen2-57B-A14B* 跳过) -------
        name_lower = model_name.lower()
        if "qwen2-57b-a14b" in name_lower:
            print("[baseline+LG] Skipped for large Qwen2-57B-A14B models.")
        else:
            print("[System] baseline + Llama Guard 2 (safeguard) ...")
            safepreds, lg_avg_time = eval_mmlu_baseline_safeguard(
                baseline_responses=base_resps,
                user_prompts=user_prompts,
                records=records,
                lg_num_gpus=args.lg_num_gpus,
                lg_max_model_len=args.lg_max_model_len,
                lg_max_new_tokens=args.lg_max_new_tokens,
                lg_batch_size=args.lg_batch_size,
            )
            safe_acc = accuracy_from_preds(records, safepreds)
            safe_total_time = base_avg_time + lg_avg_time
            print(
                f"[baseline+LG] accuracy = {safe_acc:.4f}, "
                f"main_avg_time = {base_avg_time:.4f} s/sample, "
                f"LG_avg_time = {lg_avg_time:.4f} s/sample, "
                f"total = {safe_total_time:.4f} s/sample"
            )

            summary_rows.append({
                "model": model_name,
                "model_tag": model_tag,
                "system": "baseline_safeguard_llamaguard",
                "accuracy": safe_acc,
                "main_avg_time_s": base_avg_time,
                "extra_avg_time_s": lg_avg_time,
                "total_avg_time_s": safe_total_time,
            })

        # 每个模型循环结束再保守清一次
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4) 保存 summary CSV
    summary_path = os.path.join(OUT_ROOT, "mmlu_defense_summary.csv")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[Done] Saved summary to {summary_path}")

    # 5) 按模型分别打印表格
    print("\n========== Per-model summary ==========\n")
    for model_name in summary_df["model"].unique():
        sub = summary_df[summary_df["model"] == model_name].copy()
        sub = sub[["system", "accuracy", "main_avg_time_s", "extra_avg_time_s", "total_avg_time_s"]]
        print(f"\n----- Model: {model_name} -----")
        sub_fmt = sub.copy()
        sub_fmt["accuracy"] = sub_fmt["accuracy"].map(lambda x: f"{x:.4f}")
        sub_fmt["main_avg_time_s"] = sub_fmt["main_avg_time_s"].map(lambda x: f"{x:.4f}")
        sub_fmt["extra_avg_time_s"] = sub_fmt["extra_avg_time_s"].map(lambda x: f"{x:.4f}")
        sub_fmt["total_avg_time_s"] = sub_fmt["total_avg_time_s"].map(lambda x: f"{x:.4f}")
        print(sub_fmt.to_string(index=False))


if __name__ == "__main__":
    main()
