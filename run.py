import argparse
import torch

from reticl.training.train_reticl import train_reticl
from reticl.training.pretrain_reticl import pretrain_reticl, collect_samples
from reticl.training.train_gpt2_encoder import finetune_gpt2
from reticl.training.train_bert_encoder import finetune_bert
from reticl.evaluate import evaluate, error_analysis
from reticl.analysis import visualize_representations
from reticl.data_loading.data_types import DatasetConfig
from reticl.data_loading.tabmwp import TABMWP_CONFIG
from reticl.data_loading.gsm8k import GSM8K_CONFIG
from reticl.data_loading.qasc import QASC_CONFIG
from reticl.data_loading.commonsense_qa import CQA_CONFIG
from reticl.data_loading.ag_news import AGNEWS_CONFIG
from reticl.data_loading.math import MATH_CONFIG
from reticl.data_loading.svamp import SVAMP_CONFIG
from reticl.models.generator import GeneratorCM
from reticl.constants import Datasets, RLAlgorithm, SamplingMethod, Reward, EncoderModelType, ModelType, Pooling, Init, LRSchedule
from reticl.utils import initialize_seeds, device, TrainOptions

def get_dataset_config(options_dict: dict) -> DatasetConfig:
    options = TrainOptions(options_dict)
    if options.dataset == Datasets.TABMWP.value:
        return TABMWP_CONFIG
    if options.dataset == Datasets.GSM8K.value:
        return GSM8K_CONFIG
    if options.dataset == Datasets.QASC.value:
        return QASC_CONFIG
    if options.dataset == Datasets.CQA.value:
        return CQA_CONFIG
    if options.dataset == Datasets.AGNEWS.value:
        return AGNEWS_CONFIG
    if options.dataset == Datasets.MATH.value:
        return MATH_CONFIG
    if options.dataset == Datasets.SVAMP.value:
        return SVAMP_CONFIG
    raise Exception(f"Dataset {options.dataset} not supported!")

def bool_type(arg: str):
    return False if arg == "0" else True

def main():
    if device.type == "cuda":
        if torch.cuda.device_count() > 1:
            print("Running on", torch.cuda.device_count(), "GPUs")
        else:
            print("Running on GPU")
    else:
        print("No GPU found")

    parser = argparse.ArgumentParser("RetICL")
    # Modes
    parser.add_argument("--train", action="store_true", help="Train RetICL retriever for sample lookup")
    parser.add_argument("--eval", type=str, help="Evaluate downstream performance, provide dataset split as argument")
    parser.add_argument("--create_pretrain_dataset", type=str, help="Create randomly sampled pre-training dataset, provide dataset split as argument")
    parser.add_argument("--pretrain", action="store_true", help="Pre-train model on pre-training dataset")
    parser.add_argument("--finetune_gpt2", action="store_true", help="Fine-tune GPT-2 encoder on dataset")
    parser.add_argument("--finetune_bert", action="store_true", help="Fine-tune BERT encoder on dataset")
    parser.add_argument("--error_analysis", nargs=3, help="Perform error analysis on result files; provide two .csv result files")
    parser.add_argument("--viz", action="store_true", help="Visualize retriever latent states")
    # Training options
    parser.add_argument("--dataset", type=str, choices=[dataset.value for dataset in Datasets], help="Dataset to use")
    parser.add_argument("--rl_algo", type=str, choices=[algo.value for algo in RLAlgorithm], help="RL algorithm for training")
    parser.add_argument("--sm", type=str, choices=[sm.value for sm in SamplingMethod], help="Sampling method for example retrieval")
    parser.add_argument("--early_stopping", type=bool_type, help="Enable model to select less than maximum number of examples")
    parser.add_argument("--model_type", type=str, choices=[mt.value for mt in ModelType], help="Type of RetICL model to use")
    parser.add_argument("--model_name", type=str, help="Name of RetICL model")
    parser.add_argument("--pt_model_name", type=str, help="Pre-trained model to initialize with")
    parser.add_argument("--generator_model", type=str, help="Name of pre-trained model for text generation. Set to 'gpt3' to use OpenAI, otherwise model is loaded from huggingface.", default="gpt3")
    parser.add_argument("--gpt3_model", type=str, help="Specific model when using OpenAI for generation", default="code-davinci-002")
    parser.add_argument("--gen_batch_size", type=int, help="Batch size for generator")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--wd", type=float, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, help="Gradient clipping norm total value")
    parser.add_argument("--ppo_eps", type=float, help="Epsilon value for PPO objective clipping")
    parser.add_argument("--tau", type=float, help="Off-policy: annealing coefficient for Polyak updates on critic targets")
    parser.add_argument("--replay_buffer_size", type=int, help="Off-policy: maximum size of replay buffer")
    parser.add_argument("--updates_per_batch", type=int, help="Off-policy: number of training updates per batch episodes observed")
    parser.add_argument("--train_batch_size", type=int, help="Off-policy: batch size to draw from replay buffer each training update")
    parser.add_argument("--episodes_before_train", type=int, help="Off-policy: number of episodes to observe before training")
    parser.add_argument("--init", type=str, choices=[i.value for i in Init], help="Initialization method for model parameters")
    parser.add_argument("--lr_sched", type=str, choices=[lr.value for lr in LRSchedule], help="Learning rate schedule strategy")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Number of episodes to observe per batch (equivalent to train batch size for on-policy)")
    parser.add_argument("--grad_accum_steps", type=int, help="Number of gradient accumulation steps for encoder fine-tuning")
    parser.add_argument("--num_examples", type=int, help="Number of examples to include in the prompt")
    parser.add_argument("--train_size", type=int, help="Number of samples to use for training")
    parser.add_argument("--corpus_size", type=int, help="Number of samples to use for corpus; set to 0 to use all available samples")
    parser.add_argument("--val_size", type=int, help="Number of samples to use for validation")
    parser.add_argument("--val_corpus_size", type=int, help="Number of samples to use for validation corpus; set to 0 to use all available samples")
    parser.add_argument("--save_best", type=bool_type, help="Save best model based on validation reward, otherwise save model at last epoch")
    parser.add_argument("--eg_eps", type=float, help="Initial epsilon value for epsilon-greedy sampling")
    parser.add_argument("--expl_decay_rate", type=float, help="Decay rate for exploration coefficient")
    parser.add_argument("--top_k", type=int, help="Number of top-k samples for policy approximation; set to 0 to use all samples (true policy)")
    parser.add_argument("--reward", type=str, choices=[reward.value for reward in Reward], help="Reward function")
    parser.add_argument("--encoder_model_type", type=str, choices=[emt.value for emt in EncoderModelType], help="Class of encoder model to use")
    parser.add_argument("--encoder_model", type=str, help="Pre-trained model for sample encoding")
    parser.add_argument("--encoder_h", type=int, help="Hidden size for encoder MLP; will not use MLP if not given")
    parser.add_argument("--pool", type=str, choices=[p.value for p in Pooling], help="Method for encoder final layer pooling")
    parser.add_argument("--ft_encoder", type=bool_type, help="Fine-tune encoder model during training")
    parser.add_argument("--encoder_lr", type=float, help="Learning rate for encoder fine-tuning")
    parser.add_argument("--soft_prompt_len", type=int, help="Length of encoder soft prompts")
    parser.add_argument("--hidden_size", type=int, help="Hidden size for RNN")
    parser.add_argument("--dropout", type=float, help="Dropout rate for RNN")
    parser.add_argument("--v_coef", type=float, help="Coefficient for value loss")
    parser.add_argument("--e_coef", type=float, help="Coefficient for entropy loss")
    parser.add_argument("--cr_coef", type=float, help="Coefficient for confidence reward")
    parser.add_argument("--lam", type=float, help="Lambda for GAE")
    parser.add_argument("--gamma", type=float, help="Reward discount factor")
    parser.add_argument("--sep_val_model", type=bool_type, help="Separate value model from policy model")
    parser.add_argument("--max_gen_tokens", type=int, help="Maximum number of tokens to generate")
    parser.add_argument("--beam_width", type=int, help="Beam search width for inference time policy example retrieval")
    parser.add_argument("--pt_sample_freq", type=int, help="Number of prompts to collect per sample in pretraining data")
    parser.add_argument("--rseed", type=int, help="Random seed", default=221)
    parser.add_argument("--deterministic", type=bool_type, help="Use deterministic algorithms", default=True)

    args = parser.parse_args()
    arg_dict = {arg: val for arg, val in vars(args).items() if val is not None}

    initialize_seeds(args.rseed)
    torch.use_deterministic_algorithms(args.deterministic, warn_only=True)

    dataset_config = get_dataset_config(arg_dict)
    if args.train or args.eval or args.create_pretrain_dataset:
        with GeneratorCM(arg_dict): # Load/save generator prediction cache on program start/exit
            if args.train:
                train_reticl(dataset_config, "train", "dev", arg_dict)
            if args.eval:
                evaluate(dataset_config, args.eval, arg_dict)
            if args.create_pretrain_dataset:
                collect_samples(args.create_pretrain_dataset, dataset_config, arg_dict)
    if args.pretrain:
        pretrain_reticl(dataset_config, arg_dict)
    if args.finetune_gpt2:
        finetune_gpt2(dataset_config, arg_dict)
    if args.finetune_bert:
        finetune_bert(dataset_config, arg_dict)
    if args.error_analysis:
        error_analysis(*args.error_analysis, arg_dict)
    if args.viz:
        visualize_representations(dataset_config, arg_dict)

if __name__ == "__main__":
    main()
