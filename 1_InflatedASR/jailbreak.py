import os
import json
import argparse
import warnings
import pandas as pd
from vllm import LLM, SamplingParams
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='allenai/OLMoE-1B-7B-0125-Instruct')
    parser.add_argument('--num_gpus', type=int, default=1)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    print(f'Generate Jailbreak Response by {args.model}')
    seed, max_token = 0, 1024
    llm = LLM(model=args.model, seed=seed, max_model_len=max_token, tensor_parallel_size=args.num_gpus, gpu_memory_utilization=0.8)
    sampling_params = SamplingParams(n=1, temperature=0, seed=seed, max_tokens=max_token)

    cwd = os.getcwd()
    dataset = pd.read_csv(f'{cwd}/Data/Input/jailbreaks.csv')
    get_messages = lambda queries: [[{'role':'user', 'content':q}] for q in queries]
    chats = {key:llm.chat(get_messages(dataset[key]), sampling_params, use_tqdm=False) for key in ['Original Query', 'Simplified Query']}
    responses = {k:[v.outputs[0].text for v in vs] for k, vs in chats.items()}
    json.dump(responses, open(f'{cwd}/Data/Output/{args.model.split("/")[-1]}.json', 'w'), indent=4)
    
if __name__ == '__main__':
    main()