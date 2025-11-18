#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze style-judged behaviors for MoE safety experiments.

Inputs:
  1) A CSV produced by `judge-style-behavior`:
       - example_id
       - style (including "base" and each literary style)
       - prompt
       - response
       - judge_label (REFUSE / COMPLY / UNKNOWN)
       - judge_explanation (optional)

  2) (Optional) A router JSON:
       Data/Output/safety_expert_bypass_stats.json
     to correlate Δ safety coverage with Δ refusal_rate.

  3) (Optional) A meta CSV (e.g., original jailbreaks.csv)
     with per-example metadata (category / harm_type etc.).

Outputs (by default under Data/Analysis/):
  - style_refusal_stats.csv
  - per_style_effect_vs_base.csv
  - (optional) category_style_refusal_stats.csv
  - (optional) router_behavior_merged.csv
  - several png plots (bar charts, scatter plots).
"""

import os
import argparse
import json
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------- Utils ------------------------- #

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ------------------------- 1. Load & clean ------------------------- #

def load_judge_csv(judge_csv: str) -> pd.DataFrame:
    df = pd.read_csv(judge_csv)
    required = {"example_id", "style", "prompt", "response", "judge_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Judge CSV missing required columns: {missing}")

    print("[Load] Total rows:", len(df))
    print("[Load] judge_label value_counts:")
    print(df["judge_label"].value_counts(dropna=False))

    # 丢弃 UNKNOWN，只保留 REFUSE / COMPLY
    df_clean = df[df["judge_label"].isin(["REFUSE", "COMPLY"])].copy()
    print("[Load] Clean rows (REFUSE/COMPLY only):", len(df_clean))

    return df_clean


# ------------------------- 2. Style-level refusal stats ------------------------- #

def compute_style_refusal_stats(df_clean: pd.DataFrame,
                                out_dir: str) -> pd.DataFrame:
    """
    Compute per-style refusal/comply rates and save as CSV + bar plot.

    Returns:
        style_stats: DataFrame with columns:
          style, refuse_rate, comply_rate
    """
    print("\n[StyleStats] Computing per-style refusal/comply rates...")

    # groupby style & judge_label, compute normalized counts
    style_stats = (
        df_clean
        .groupby("style")["judge_label"]
        .value_counts(normalize=True)
        .rename("ratio")
        .reset_index()
        .pivot(index="style", columns="judge_label", values="ratio")
        .fillna(0.0)
        .reset_index()
    )

    # Make sure we have both columns
    if "REFUSE" not in style_stats.columns:
        style_stats["REFUSE"] = 0.0
    if "COMPLY" not in style_stats.columns:
        style_stats["COMPLY"] = 0.0

    style_stats.rename(columns={"REFUSE": "refuse_rate",
                                "COMPLY": "comply_rate"},
                       inplace=True)

    style_stats = style_stats.sort_values("refuse_rate", ascending=False)
    print(style_stats)

    # Save CSV
    out_csv = os.path.join(out_dir, "style_refusal_stats.csv")
    style_stats.to_csv(out_csv, index=False)
    print(f"[StyleStats] Saved style refusal stats to {out_csv}")

    # Bar plot of refusal_rate
    plt.figure(figsize=(10, 5))
    plt.bar(style_stats["style"], style_stats["refuse_rate"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Refusal rate (judge_label == REFUSE)")
    plt.title("Refusal rate per style")
    plt.tight_layout()
    out_png = os.path.join(out_dir, "style_refusal_rates.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[StyleStats] Saved bar plot to {out_png}")

    return style_stats


# ------------------------- 3. Example-level base vs style effect ------------------------- #

def compute_per_example_effect(df_clean: pd.DataFrame,
                               out_dir: str) -> pd.DataFrame:
    """
    For each example, compare base vs each style:
      - improved: base=COMPLY & style=REFUSE
      - degraded: base=REFUSE & style=COMPLY

    Returns a DataFrame with per-style:
      style, improved_rate_overall, degraded_rate_overall,
      improved_rate_given_base_comply
    """
    print("\n[Effect] Computing per-example base vs style effects...")

    # pivot: each example_id is a row; each style a column with judge_label
    small = df_clean[["example_id", "style", "judge_label"]].copy()
    pivot = small.pivot(index="example_id", columns="style", values="judge_label")

    if "base" not in pivot.columns:
        raise ValueError("Column 'base' not found in style column; need a 'base' style as reference.")

    style_names = [c for c in pivot.columns if c != "base"]
    base_labels = pivot["base"]

    results = []

    for style in style_names:
        style_labels = pivot[style]

        # only consider rows where both base and style labels exist
        mask_valid = base_labels.notna() & style_labels.notna()
        base_valid = base_labels[mask_valid]
        style_valid = style_labels[mask_valid]

        if len(base_valid) == 0:
            print(f"[Effect] Style {style}: no valid pairs with base; skipping")
            continue

        # 1) improved: base=COMPLY -> style=REFUSE
        improved_overall = ((base_valid == "COMPLY") & (style_valid == "REFUSE")).mean()

        # 2) degraded: base=REFUSE -> style=COMPLY
        degraded_overall = ((base_valid == "REFUSE") & (style_valid == "COMPLY")).mean()

        # 3) conditional: among base=COMPLY, how often style=REFUSE?
        mask_base_comply = (base_valid == "COMPLY")
        if mask_base_comply.any():
            improved_cond = ((style_valid == "REFUSE") & mask_base_comply).sum() / mask_base_comply.sum()
        else:
            improved_cond = 0.0

        results.append({
            "style": style,
            "improved_rate_overall": float(improved_overall),
            "degraded_rate_overall": float(degraded_overall),
            "improved_rate_given_base_comply": float(improved_cond),
            "num_valid_pairs": int(mask_valid.sum()),
        })

    per_style_effect = pd.DataFrame(results).sort_values(
        "improved_rate_given_base_comply", ascending=False
    )

    print(per_style_effect)

    out_csv = os.path.join(out_dir, "per_style_effect_vs_base.csv")
    per_style_effect.to_csv(out_csv, index=False)
    print(f"[Effect] Saved per-style effect vs base to {out_csv}")

    return per_style_effect


# ------------------------- 4. Optional: Category × style stats ------------------------- #

def compute_category_style_stats(df_clean: pd.DataFrame,
                                 meta_csv: str,
                                 category_col: str,
                                 out_dir: str) -> Optional[pd.DataFrame]:
    """
    If meta CSV and a category column are provided, compute refusal/comply
    rates per (category, style).

    meta_csv is assumed to be the original jailbreak CSV, whose row index
    corresponds to example_id. We add an 'example_id' column so that we can
    merge with df_clean on example_id.

    Returns the stats DataFrame, or None if something fails.
    """
    print("\n[Category] Computing (category, style) refusal stats...")

    if not os.path.exists(meta_csv):
        print(f"[Category] meta_csv not found: {meta_csv}, skip.")
        return None

    meta = pd.read_csv(meta_csv)
    meta["example_id"] = meta.index

    if category_col not in meta.columns:
        print(f"[Category] Column '{category_col}' not in meta_csv; skip.")
        return None

    df_merged = df_clean.merge(
        meta[["example_id", category_col]], on="example_id", how="left"
    )

    if df_merged[category_col].isna().all():
        print("[Category] All category values are NaN, skip.")
        return None

    cat_style_stats = (
        df_merged[df_merged["judge_label"].isin(["REFUSE", "COMPLY"])]
        .groupby([category_col, "style"])["judge_label"]
        .value_counts(normalize=True)
        .rename("ratio")
        .reset_index()
        .pivot(index=[category_col, "style"],
               columns="judge_label",
               values="ratio")
        .fillna(0.0)
        .reset_index()
    )

    if "REFUSE" not in cat_style_stats.columns:
        cat_style_stats["REFUSE"] = 0.0
    if "COMPLY" not in cat_style_stats.columns:
        cat_style_stats["COMPLY"] = 0.0

    cat_style_stats.rename(columns={"REFUSE": "refuse_rate",
                                    "COMPLY": "comply_rate"},
                           inplace=True)

    out_csv = os.path.join(out_dir, "category_style_refusal_stats.csv")
    cat_style_stats.to_csv(out_csv, index=False)
    print(f"[Category] Saved category×style refusal stats to {out_csv}")

    return cat_style_stats


# ------------------------- 5. Optional: Merge with router stats ------------------------- #

def merge_with_router_stats(style_stats: pd.DataFrame,
                            router_json: str,
                            out_dir: str) -> Optional[pd.DataFrame]:
    """
    Merge style-level behavior stats (refuse_rate) with router-level
    stats (mean_delta_coverage, bypass_rate_structural).

    style_stats:
      must contain columns: 'style', 'refuse_rate'

    router_json:
      produced by analyze-bypass, with structure:
        {
          "bypass_threshold": ...,
          "percentile": ...,
          "mean_base_coverage": ...,
          "styles": {
              "shakespeare": {
                  "mean_style_coverage": ...,
                  "mean_delta": ...,
                  "bypass_rate": ...
              },
              ...
          }
        }

    Writes:
      router_behavior_merged.csv
      scatter plot: coverage_vs_refuse_delta.png
    """
    if not os.path.exists(router_json):
        print(f"[RouterMerge] router_json not found: {router_json}, skip.")
        return None

    with open(router_json, "r", encoding="utf-8") as f:
        router_stats = json.load(f)

    mean_base_cover = router_stats.get("mean_base_coverage", None)
    print(f"\n[RouterMerge] mean_base_coverage = {mean_base_cover}")

    rows = []
    for style, stat in router_stats.get("styles", {}).items():
        rows.append({
            "style": style,
            "mean_delta_coverage": stat.get("mean_delta", 0.0),
            "bypass_rate_structural": stat.get("bypass_rate", 0.0),
        })
    router_df = pd.DataFrame(rows)

    # style_stats has 'style' and 'refuse_rate'
    if "style" not in style_stats.columns or "refuse_rate" not in style_stats.columns:
        print("[RouterMerge] style_stats missing 'style' or 'refuse_rate'; skip merge.")
        return None

    # base refusal rate
    if "base" not in list(style_stats["style"]):
        print("[RouterMerge] base style not found in style_stats; skip merge.")
        return None

    base_refuse = float(
        style_stats.loc[style_stats["style"] == "base", "refuse_rate"].iloc[0]
    )
    print(f"[RouterMerge] base refusal_rate = {base_refuse:.4f}")

    tmp = style_stats.copy()
    tmp["delta_refuse_rate"] = tmp["refuse_rate"] - base_refuse

    merged = router_df.merge(tmp[["style", "refuse_rate", "delta_refuse_rate"]],
                             on="style", how="inner")

    print("\n[RouterMerge] Merged router + behavior stats:")
    print(merged)

    # correlation
    if len(merged) > 1:
        corr = np.corrcoef(merged["mean_delta_coverage"],
                           merged["delta_refuse_rate"])[0, 1]
        print(f"[RouterMerge] corr(mean_delta_coverage, delta_refuse_rate) = {corr:.4f}")
    else:
        print("[RouterMerge] Not enough styles for correlation.")

    out_csv = os.path.join(out_dir, "router_behavior_merged.csv")
    merged.to_csv(out_csv, index=False)
    print(f"[RouterMerge] Saved merged stats to {out_csv}")

    # Scatter plot: x = mean_delta_coverage, y = delta_refuse_rate
    plt.figure(figsize=(6, 5))
    plt.scatter(merged["mean_delta_coverage"], merged["delta_refuse_rate"])
    for x, y, name in zip(merged["mean_delta_coverage"],
                          merged["delta_refuse_rate"],
                          merged["style"]):
        plt.text(x, y, name, fontsize=8, ha="center", va="bottom")

    plt.axvline(0.0, linestyle="--")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Mean Δ safety coverage (style - base)")
    plt.ylabel("Δ refusal rate (style - base)")
    plt.title("Structural vs behavioral change per style")
    plt.tight_layout()
    out_png = os.path.join(out_dir, "coverage_vs_refuse_delta.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[RouterMerge] Saved scatter plot to {out_png}")

    return merged


# ------------------------- Main ------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Analyze judged style behaviors for MoE safety experiments."
    )
    parser.add_argument(
        "--judge_csv", type=str, required=True,
        help="CSV produced by judge-style-behavior (example_id, style, prompt, response, judge_label...)."
    )
    parser.add_argument(
        "--out_dir", type=str, default="Data/Analysis",
        help="Directory to save analysis outputs (CSVs + plots)."
    )

    # Optional: meta CSV for category-level analysis
    parser.add_argument(
        "--meta_csv", type=str, default=None,
        help="(Optional) meta CSV (e.g., original jailbreaks.csv) whose row index is example_id."
    )
    parser.add_argument(
        "--meta_category_col", type=str, default=None,
        help="(Optional) column name in meta_csv used as harmful category / type."
    )

    # Optional: router JSON for structure vs behavior correlation
    parser.add_argument(
        "--router_json", type=str, default=None,
        help="(Optional) JSON from analyze-bypass with safety_expert stats."
    )

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # 1) Load & clean
    df_clean = load_judge_csv(args.judge_csv)

    # 2) Style-level refusal/comply stats
    style_stats = compute_style_refusal_stats(df_clean, args.out_dir)

    # 3) Example-level base vs style effect
    per_style_effect = compute_per_example_effect(df_clean, args.out_dir)

    # 4) Optional: category × style
    if args.meta_csv and args.meta_category_col:
        compute_category_style_stats(
            df_clean, args.meta_csv, args.meta_category_col, args.out_dir
        )
    else:
        print("\n[Category] meta_csv or meta_category_col not provided; skip category-level analysis.")

    # 5) Optional: merge with router stats
    if args.router_json:
        merge_with_router_stats(style_stats, args.router_json, args.out_dir)
    else:
        print("\n[RouterMerge] router_json not provided; skip router-behavior correlation.")

    print("\n[Done] Analysis finished.")


if __name__ == "__main__":
    main()
