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