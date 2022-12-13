# RetICL
RetICL (Retrieval for In-Context Learning) is a reinforcement learning-based method for the joint retrieval of in-context learning samples. The primary component of the method is a recurrent neural network that represents a group of examples along with a problem's context, and is able to rank candidate examples to be added to the group. The most relevant code files are `models/retriever.py`, `data_loading/data_loading.py`, and `training.py`.

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

For GPT-3 and Codex models, you need an API key from https://openai.com/api/. Set it in your environment:
```
export OPENAI_API_KEY_1="<your key>"
```

### Data

Download Dataset: https://github.com/lupantech/PromptPG/tree/main/data/tabmwp

Make `tabmwp` folder accessible from this repo's root folder.

## Run

You can see all options by running `python3 run.py --help`. Default values can be found in the `TrainOptions` constructor in `utils.py`.

Train example:
```
python3 run.py --train --method mcc --model_name ret_mcc
```

Test example:
```
python3 run.py --eval dev1k --method mcc --model_name ret_mcc
```

## External Code

Please note that files in the `promptPG` folder are copied from the repo https://github.com/lupantech/PromptPG. These files exclusively relate to data processing. The associated license is copied to `promptPG/LICENSE.md`.
