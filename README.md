# RetICL
RetICL (Retrieval for In-Context Learning) is a reinforcement learning-based method for the joint retrieval of in-context learning examples. The primary component is a recurrent neural network that jointly represents a problem and a group of examples, along with a bilinear activation that ranks subsequent examples.

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

Place those folders in this repo's root folder.

## Run

You can see all options by running `python3 run.py --help`. Default values can be found in the `TrainOptions` constructor in `utils.py`.

Train example:
```
python3 run.py --train --rl_algo ppo --sm softmax --dataset gsm8k --model_name gsm8k_ppo
```

Test example:
```
python3 run.py --eval dev1k --rl_algo ppo --sm softmax --dataset gsm8k --model_name gsm8k_ppo
```

## External Code

Please note that files in the `promptPG` folder are copied from the repo https://github.com/lupantech/PromptPG. These files exclusively relate to data processing. The associated license is copied to `promptPG/LICENSE.md`.
