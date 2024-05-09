# RetICL
[RetICL (Retrieval for In-Context Learning)](https://arxiv.org/abs/2305.14502) is a reinforcement learning-based method for the joint retrieval of in-context learning examples. The primary component is a recurrent neural network that jointly represents a problem and a group of examples, along with a bilinear activation that ranks subsequent examples. We also introduce a confidence-based reward where the perplexity of the generated solution is used as a proxy for the quality of the reasoning.

## Citation
If you found this code or these ideas useful, please cite our paper!
```
@misc{scarlatos2024reticl,
      title={RetICL: Sequential Retrieval of In-Context Examples with Reinforcement Learning}, 
      author={Alexander Scarlatos and Andrew Lan},
      year={2024},
      eprint={2305.14502},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Setup

### Python Environment
Ensure you have Python3 installed (this code was tested on v3.9.1).

Create and activate a virtual environment:
```
python3 -m venv <env_name>
source <env_name>/bin/activate
```

Install dependencies with pip:
```
python3 -m pip install -r requirements.txt
```

### OpenAI API Keys

For GPT-3 and Codex models, you need an API key(s) from https://openai.com/api/. Set it in your environment:
```
export OPENAI_API_KEYS="<your key 1>,<your key 2>,..."
```

### Data

TabMWP: https://github.com/lupantech/PromptPG/tree/main/data/tabmwp

GSM8K: https://github.com/openai/grade-school-math

QASC: http://data.allenai.org/downloads/qasc/qasc_dataset.tar.gz

CommonsenseQA: https://www.tau-nlp.sites.tau.ac.il/commonsenseqa

ECQA: https://github.com/dair-iitd/ECQA-Dataset

SVAMP: https://github.com/arkilpatel/SVAMP

Place those folders in this repo's root folder.

## Run

You can see all options by running `python3 run.py --help`. Default values can be found in the `TrainOptions` constructor in `utils.py`.


### Examples

Train:
```
python3 run.py --train --rl_algo ppo_simple --dataset gsm8k --model_name gsm8k_ppo --e_coef .1 --train_size 5000 --corpus_size 200 --soft_prompt_len 20 --val_size 500 --wandb
```

Test:
```
python3 run.py --eval test --rl_algo ppo_simple --dataset gsm8k --model_name gsm8k_ppo --soft_prompt_len 20 --wandb
```

Baselines:
```
python3 run.py --eval test --sm random --dataset gsm8k
python3 run.py --eval test --sm sim --dataset gsm8k
python3 run.py --eval test --sm complex --dataset gsm8k
```

LSTM Classifier Baseline:
```
python3 run.py --create_pretrain_dataset --pt_sample_freq 20 --dataset gsm8k
python3 run.py --pretrain --pt_sample_freq 20 --dataset gsm8k --pt_model_name gsm8k_lstm_classifier
python3 run.py --eval test --sm vf --dataset gsm8k --model_name gsm8k_lstm_classifier
```

## External Code

Files in the `promptPG` folder are copied from the repo https://github.com/lupantech/PromptPG. The associated license is copied to `promptPG/LICENSE.md`.
