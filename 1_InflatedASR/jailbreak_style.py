import argparse
import json
import os
import warnings

import pandas as pd
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams


def run_chat_batch(llm: LLM, texts, sampling_params: SamplingParams):
    """Run llm.chat on a list of texts and return a list of responses."""

    def get_messages(queries):
        return [[{"role": "user", "content": q if isinstance(q, str) else ""}]
                for q in queries]

    messages = get_messages(texts)
    outputs = llm.chat(messages, sampling_params, use_tqdm=True)
    responses = [o.outputs[0].text for o in outputs]
    return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMoE-1B-7B-0125-Instruct",
        help="Model name for evaluation.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="Data/Input/jailbreaks_literary.csv",
        help="CSV file with styled query columns (e.g. 'shakespeare Query').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Data/Output",
        help="Directory to save per-style JSON files.",
    )
    parser.add_argument(
        "--styles",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional list of style names to eval "
            "(e.g. shakespeare opera). If not set, auto-detect from columns."
        ),
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    print(f"Generate jailbreak responses by {args.model}")

    seed, max_token = 0, 1024
    llm = LLM(
        model=args.model,
        seed=seed,
        max_model_len=max_token,
        tensor_parallel_size=args.num_gpus,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,  # deterministic
        top_p=1.0,
        max_tokens=max_token,
    )

    cwd = os.getcwd()
    input_path = os.path.join(cwd, args.input_csv)
    df = pd.read_csv(input_path)

    # Auto-detect style columns if not explicitly provided:
    #   columns like "shakespeare Query", "opera Query", ...
    if args.styles is None:
        style_cols = []
        for col in df.columns:
            # We assume style columns follow "<style_name> Query"
            if col.endswith(" Query") and col not in ["Original Query", "Simplified Query"]:
                style_cols.append(col)
        if not style_cols:
            raise ValueError(
                "No style columns found. Expect columns like 'shakespeare Query', 'opera Query', etc."
            )
        # Map to style names (everything before ' Query')
        styles = [c[:-len(" Query")] for c in style_cols]
    else:
        # Use user-provided style names; build column names from them
        styles = args.styles
        style_cols = [f"{s} Query" for s in styles]

    print("[Info] Styles to evaluate:", styles)

    os.makedirs(os.path.join(cwd, args.output_dir), exist_ok=True)
    model_tag = args.model.split("/")[-1]

    # Loop over each style and write a separate JSON
    for style_name, col_name in tqdm(
            list(zip(styles, style_cols)),
            desc="Evaluating styles"
    ):
        if col_name not in df.columns:
            raise ValueError(
                f"Column '{col_name}' not found in CSV. "
                f"Expected a column named '{col_name}' for style '{style_name}'."
            )

        print(f"[Info] Evaluating style: {style_name} (column: '{col_name}')")
        queries = df[col_name].fillna("").tolist()

        responses = run_chat_batch(llm, queries, sampling_params)

        # Pack into a JSON structure – you can adjust keys if you want
        result = {
            "model": args.model,
            "style": style_name,
            "column": col_name,
            "queries": queries,
            "responses": responses,
        }

        out_path = os.path.join(
            cwd,
            args.output_dir,
            f"{model_tag}_{style_name}.json",
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        print(f"[Info] Saved {style_name} results to {out_path}")


if __name__ == "__main__":
    main()
