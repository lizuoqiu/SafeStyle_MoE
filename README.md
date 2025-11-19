# Style induced MoE Jailbreak
## Environment Set up
1. install conda through: https://www.anaconda.com/docs/getting-started/miniconda/install
2. accept conda TOS:  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main  and conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
 to create the conda environment
3. create conda environment moe_attack
4. install pytorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
5. install requirements through pip install -r requirements.txt
6. accept hugging face API access through  huggingface-cli login
7. regenerate your OpenAI API key


# MoE Safety Routing with Style-Induced Jailbreaks

This repo provides a pipeline to study **style-induced routing and safety-expert bypass** in Mixture-of-Experts (MoE) language models.

We:

1. Build a **calibration set** of *refusal* vs *comply* examples for a given MoE model.
2. Extract **router distributions** on harmful base prompts and on their **literary / stylistic variants**.
3. Identify **safety-critical experts** and quantify **bypass rates** under different styles.
4. Generate base + style responses with vLLM.
5. Use a local / vLLM **LLM Judge** to label each (prompt, response) as `REFUSE` or `COMPLY`.

All outputs are organized **per tested model** (e.g., `allenai/OLMoE-1B-7B-0924-Instruct`) so you can run many models and keep their artifacts separate.

---

## Directory Layout

By default, the script will create per-model subdirectories based on the HF model name:

- `Data/Calib/<model_tag>/`
 - `calibration_all.csv`
 - `refuse.csv`
 - `comply.csv`

- `Data/Router/<model_tag>/`
 - `safe_router.npy`
 - `unsafe_router.npy`
 - `base_router.npy`
 - `style_shakespeare.npy`, `style_biblical.npy`, ...

- `Data/Output/<model_tag>/`
 - `style_generations.csv`
 - `style_generations_judged.csv`
 - `style_generations_judged_vllm.csv`
 - `safety_expert_bypass_stats.json`
 - `style_safety_bypass_scatter.png`

Here `<model_tag>` is a sanitized version of the model name:
`allenai/OLMoE-1B-7B-0924-Instruct` → `allenai_OLMoE-1B-7B-0924-Instruct`
(i.e., `/` and `:` are replaced by `_`).

---

## Pipeline Overview

We expose the following sub-commands:

1. `prepare-calib`  
   Run the MoE model on a harmful dataset; split into *refusal* vs *comply* via regex and save CSVs.

2. `dump-router`  
   Compute per-sample router probability tensors for:
 - calibration safe (refuse) prompts
 - calibration unsafe (comply) prompts
 - base jailbreak queries
 - style-transformed jailbreak queries

3. `analyze-bypass`  
   Use calibration router stats to identify **safety experts**, then compute:
 - mean safety-expert coverage for base vs style
 - **bypass rate** (fraction of samples whose coverage significantly drops under style prompts)

4. `prepare-style-behavior`  
   Use **vLLM** to generate model responses for:
 - base jailbreak queries
 - each style-transformed jailbreak query  
   and save a long CSV `(example_id, style, prompt, response)`.

5. `judge-style-behavior` (HF)  
   Use a local HF model as **judge** to label each `(prompt, response)` as `REFUSE` or `COMPLY`.

6. `judge-style-behavior-vllm` (vLLM)  
   Same as above, but the judge runs via **vLLM** for higher throughput.

---

## Example: OLMoE as Target Model

Assume:

- Target MoE model:  
  `allenai/OLMoE-1B-7B-0924-Instruct`
- Judge model (HF or vLLM):  
  `allenai/OLMoE-1B-7B-0125-Instruct`
- Harmful dataset CSV:  
  `Data/Harmbench/harmbench_prompts.csv` with column `Original Query`
- Jailbreak + style prompt CSV:  
  `Data/Jailbreak/style_queries.csv` with columns:
 - `Original Query` (base harmful prompt)
 - `shakespeare Query`, `biblical Query`, ..., `mythological Query` (12 styles)

The main script is assumed to be:

```bash
python moe_safety_routing_pipeline_with_LLM_Judge.py 
```

# Step 1 — Calibration: prepare-calib

Run the target MoE model on a harmful dataset and split into “refuse” vs “comply” using a simple regex heuristic:

```
python moe_safety_routing_pipeline_with_LLM_Judge.py \
  prepare-calib \
  --model allenai/OLMoE-1B-7B-0125-Instruct \
  --harm_csv Data/Harmbench/harmbench_prompts.csv \
  --prompt_column "Original Query" \
  --out_dir Data/Calib \
  --max_tokens 256 \
  --batch_size 8 \
  --max_length 512

```

# Step 2 — Dump Router Arrays: dump-router

Collect router probabilities for:

- Calibration “refuse” prompts → safe_router.npy

- Calibration “comply” prompts → unsafe_router.npy

- Base jailbreak prompts → base_router.npy

- Style jailbreak prompts → style_<style>.npy


Run:

```
python moe_safety_routing_pipeline_with_LLM_Judge.py \
  dump-router \
  --model allenai/OLMoE-1B-7B-0924-Instruct \
  --calib_dir Data/Calib \
  --jailbreak_csv Data/Jailbreak/style_queries.csv \
  --base_column "Original Query" \
  --router_dir Data/Router \
  --batch_size 8 \
  --max_length 512

```

To restrict to a subset of styles (e.g., Shakespeare + Scientific):

```
python moe_safety_routing_pipeline_with_LLM_Judge.py \
  dump-router \
  --model allenai/OLMoE-1B-7B-0924-Instruct \
  --calib_dir Data/Calib \
  --jailbreak_csv Data/Jailbreak/style_queries.csv \
  --base_column "Original Query" \
  --router_dir Data/Router \
  --styles shakespeare scientific \
  --batch_size 8 \
  --max_length 512
```

Outputs in Data/Router/:

- safe_router.npy # refusals

- unsafe_router.npy # complies

- base_router.npy # base jailbreaks

- style_<style>.npy # one per style


# Step 3 — Analyze Safety-Expert Bypass: analyze-bypass

Here we:

1. Compute per-expert safety scores:
$ score(layer, expert) = mean_prob_refuse − mean_prob_comply $

2. Mark top-percentile experts per layer as safety-critical experts.

3. Compute, for each sample, how much probability mass goes to these safety experts.

4. Compare base vs styled prompts:

$ Mean Δ coverage (style − base) $

Bypass rate: fraction of cases where safety coverage drops below a threshold.

Run:

```
python moe_safety_routing_pipeline_with_LLM_Judge.py \
  analyze-bypass \
  --router_dir Data/Router/allenai_OLMoE-1B-7B-0924-Instruct \
  --output_json Data/Output/safety_expert_bypass_stats.json \
  --bypass_threshold 0.05 \
  --percentile 0.8

```

Outputs:

- Data/Output/safety_expert_bypass_stats.json

- Global mean base coverage

- Per-style mean Δ coverage

- Per-style bypass rate

- Data/Output/style_safety_bypass_scatter.png

- Scatter plot: x = mean Δ coverage, y = bypass rate per style


