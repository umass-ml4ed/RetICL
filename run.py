import argparse
import torch

from training.reticl import train_reticl
from training.gpt2_encoder import finetune_gpt2
from training.bert_encoder import finetune_bert
from evaluate import evaluate_reticl, error_analysis
from analysis import visualize_representations
from models.generator import GeneratorCM
from constants import Datasets, RLAlgorithm, SamplingMethod, Reward, EncoderModelType
from utils import initialize_seeds, device

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
    parser.add_argument("--finetune_gpt2", action="store_true", help="Fine-tune GPT-2 encoder on dataset")
    parser.add_argument("--finetune_bert", action="store_true", help="Fine-tune BERT encoder on dataset")
    parser.add_argument("--error_analysis", nargs=3, help="Perform error analysis on result files; provide group to analyze followed by two .csv result files")
    parser.add_argument("--viz", action="store_true", help="Visualize retriever latent states")
    # Training options
    parser.add_argument("--dataset", type=str, choices=[dataset.value for dataset in Datasets], help="Dataset to use")
    parser.add_argument("--rl_algo", type=str, choices=[algo.value for algo in RLAlgorithm], help="RL algorithm for training")
    parser.add_argument("--sm", type=str, choices=[sm.value for sm in SamplingMethod], help="Sampling method for example retrieval")
    parser.add_argument("--model_name", type=str, help="Name of retriever model")
    parser.add_argument("--generator_model", type=str, help="Name of pre-trained model for text generation", default="gpt3") # "EleutherAI/gpt-j-6B"
    parser.add_argument("--gpt3_model", type=str, help="Specific model when using GPT-3 for generation", default="code-davinci-002")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--baseline", action="store_true", help="Use baseline (example-independent) model")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--wd", type=float, help="Weight decay")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--grad_accum_steps", type=int, help="Number of gradient accumulation steps for encoder fine-tuning")
    parser.add_argument("--num_examples", type=int, help="Number of examples to include in the prompt")
    parser.add_argument("--train_size", type=int, help="Number of samples to use for training")
    parser.add_argument("--corpus_size", type=int, help="Number of samples to use for corpus; set to 0 to use all available samples")
    parser.add_argument("--epsilon", type=float, help="Initial value for epsilon-greedy sampling")
    parser.add_argument("--epsilon_decay", type=float, help="Decay rate for epsilon")
    parser.add_argument("--top_k", type=int, help="Number of top-k samples for policy approximation; set to 0 to use all samples (true policy)")
    parser.add_argument("--reward", type=str, choices=[reward.value for reward in Reward], help="Reward function")
    parser.add_argument("--encoder_model_type", type=str, choices=[emt.value for emt in EncoderModelType], help="Class of encoder model to use")
    parser.add_argument("--encoder_model", type=str, help="Pre-trained model for sample encoding")
    parser.add_argument("--hidden_size", type=int, help="Hidden size for RNN")
    parser.add_argument("--dropout", type=float, help="Dropout rate for RNN")
    parser.add_argument("--v_coef", type=float, help="Coefficient for value loss")
    parser.add_argument("--e_coef", type=float, help="Coefficient for entropy loss")
    parser.add_argument("--rseed", type=int, help="Random seed", default=221)

    args = parser.parse_args()
    arg_dict = {arg: val for arg, val in vars(args).items() if val is not None}

    initialize_seeds(args.rseed)

    if args.train or args.eval:
        with GeneratorCM(arg_dict): # Load/save generator prediction cache on program start/exit
            if args.train:
                train_reticl(arg_dict)
            if args.eval:
                evaluate_reticl(arg_dict, args.eval)
    if args.finetune_gpt2:
        finetune_gpt2(arg_dict)
    if args.finetune_bert:
        finetune_bert(arg_dict)
    if args.error_analysis:
        error_analysis(*args.error_analysis, arg_dict)
    if args.viz:
        visualize_representations(arg_dict)

if __name__ == "__main__":
    main()
