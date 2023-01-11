import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import nltk

from models.retriever import Retriever
from models.generator import Generator
from data_loading.data_loading import Collator, CollatedBatch
from data_loading.tabmwp import TabMWPDataset
from data_loading.gsm8k import GSM8KDataset
from evaluate import check_correct, evaluate
from constants import Datasets, SamplingMethod, Reward
from utils import TrainOptions, device

def get_rewards(batch: CollatedBatch, options: TrainOptions):
    if options.reward in (Reward.EXACT.value, Reward.EXACT_AND_BLEU.value):
        # Generate predictions given retrieved context and check correctness
        predictions = Generator.generate(**batch)
        correct = torch.Tensor([
            1 if check_correct(src_meta_data, pred, options) else 0
            for src_meta_data, pred in zip(batch["meta_data"], predictions)
        ])

        if options.reward == Reward.EXACT.value:
            # Reward is 1 if prediction is correct, -1 otherwise
            return 2 * correct - 1

        if options.reward == Reward.EXACT_AND_BLEU.value:
            # Calculate bleu score on the generated solutions
            bleu = torch.Tensor([
                nltk.translate.bleu([target], pred)
                for pred, target in zip (predictions, batch["labels"])
            ])
            # Half of reward comes from bleu and other half from final correctness
            return correct + bleu - 1

    # Reward is negative perplexity assigned to label given the context
    if options.reward == Reward.PPL.value:
        return -Generator.get_ppl(**batch)

def get_td_error(value_estimates: torch.Tensor, rewards: torch.Tensor):
    batch_size = rewards.shape[0]
    max_num_examples = value_estimates.shape[0] // batch_size
    # Append 0 to value estimates for terminal state
    v_t = F.pad(value_estimates.detach().view(batch_size, -1), (0, 1))
    # Per-example reward is 0 except for final example
    r_t = F.pad(rewards.unsqueeze(1), (max_num_examples - 1, 0))
    # TD error: r_t + v_(t+1) - v_t
    return (r_t + v_t[:, 1:] - v_t[:, :-1]).view(-1)

def train_retriever(options_dict: dict):
    options = TrainOptions(options_dict)
    if options.wandb:
        run = wandb.init(project="reticl", config=options.as_dict())
    else:
        run = None
    retriever = Retriever(options).to(device)
    retriever.train()
    if options.dataset == Datasets.TABMWP.value:
        dataset = TabMWPDataset("train", retriever, options)
        val_set = TabMWPDataset("dev500", retriever, options)
    elif options.dataset == Datasets.GSM8K.value:
        dataset = GSM8KDataset("train", retriever, options)
        val_set = GSM8KDataset("dev500", retriever, options)
    else:
        raise Exception(f"Dataset {options.dataset} not supported!")
    val_set.set_greedy(True) # Use greedy retrieval for validation
    data_loader = DataLoader(
        dataset,
        collate_fn=Collator(),
        batch_size=options.batch_size,
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_set,
        collate_fn=Collator(),
        batch_size=options.batch_size,
        shuffle=False,
        drop_last=False
    )
    optimizer = torch.optim.AdamW(retriever.parameters(), lr=options.lr)
    torch.autograd.set_detect_anomaly(True)
    print("Training...")
    previous_model = None # For PPO
    best_model = Retriever(options).to(device)
    best_reward = None
    for epoch in range(options.epochs):
        total_reward = 0
        total_loss = 0

        # Update epsilon if we're doing epsilon-greedy sampling
        if options.method == SamplingMethod.MCC.value:
            dataset.update_epsilon(options.epsilon * options.epsilon_decay ** epoch)
            print("Epsilon:", dataset.epsilon)

        # Sample batch from dataset - example retrieval is also done here (__getitem__ in DatasetBase)
        for batch in tqdm(data_loader):
            batch_size, max_num_examples = batch["example_encodings"].shape[:2]
            optimizer.zero_grad()

            # Calculate rewards for batch
            rewards = get_rewards(batch, options).to(device)

            # Calculate returns: gamma=1 so just repeat reward over each time step
            returns = rewards.unsqueeze(1).repeat(1, max_num_examples).view(-1) # (N * L)

            # Get activations on current examples and value function estimates from retriever
            activations, value_estimates = retriever(**batch)

            # Calculate loss and backprop
            loss_mask = torch.arange(max_num_examples).expand(batch_size, -1) >= batch["num_examples"].unsqueeze(1)
            loss_mask = loss_mask.view(-1)
            if options.method == SamplingMethod.MCC.value:
                loss_fn = torch.nn.MSELoss(reduction="none")
                loss = loss_fn(activations, returns)
            elif options.method == SamplingMethod.PG.value:
                # REINFORCE: param = param + lr * G * grad(log(pi[a]))
                # GD: param = param - lr * grad(loss)
                # loss = -G * log(pi[a])
                # pi[a] = softmax(activations)[a]
                # CEL = -log(softmax(activations)[a])
                # loss = G * CEL
                loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fn(activations, batch["policy_example_indices"].view(-1))
                loss = loss * returns.view(-1)
            elif options.method == SamplingMethod.RWB.value:
                pg_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                pg_loss = pg_loss_fn(activations, batch["policy_example_indices"].view(-1))
                pg_loss = pg_loss * (returns.view(-1) - value_estimates.detach()) # Don't differentiate w.r.t. baseline
                val_loss = torch.nn.MSELoss(reduction="none")(value_estimates, returns)
                loss = pg_loss + options.v_coef * val_loss
            elif options.method == SamplingMethod.AC.value:
                td_error = get_td_error(value_estimates, rewards)
                pg_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                pg_loss = pg_loss_fn(activations, batch["policy_example_indices"].view(-1))
                pg_loss = pg_loss * td_error
                # (r_t + v_(t+1) - v_t)^2 = ((r_t + v_(t+1) - v_t + v_t) - v_t)^2 = ((td_err + v_t) - v_t)^2
                val_loss = torch.nn.MSELoss(reduction="none")(value_estimates, td_error + value_estimates.detach())
                loss = pg_loss + options.v_coef * val_loss
            elif options.method == SamplingMethod.PPO.value:
                # Get policy ratio
                if not previous_model:
                    ratio = torch.ones((batch_size * max_num_examples)).to(device)
                    previous_model = Retriever(options).to(device)
                else:
                    with torch.no_grad():
                        pi_old_activations, _ = previous_model(**batch)
                    cur_policy_probs = torch.softmax(activations, dim=-1)[torch.arange(batch_size * max_num_examples), batch["policy_example_indices"].view(-1)]
                    old_policy_probs = torch.softmax(pi_old_activations, dim=-1)[torch.arange(batch_size * max_num_examples), batch["policy_example_indices"].view(-1)]
                    ratio = cur_policy_probs / old_policy_probs
                # Copy model for next iteration
                previous_model.load_state_dict(retriever.state_dict())

                # Get estimated advantage - using one-step TD error
                advantage = get_td_error(value_estimates, rewards)

                # Get clip loss
                clip_loss = -torch.min(ratio * advantage, torch.clip(ratio, 0.8, 1.2) * advantage)

                # Get value function loss
                val_loss = torch.nn.MSELoss(reduction="none")(value_estimates, returns)

                # Get entropy loss - encourages exploration by flattening action distribution
                # entropy_loss = -torch.log(torch.softmax(activations, dim=-1).clip(1e-35)).mean()

                # Get final loss
                # loss = clip_loss + options.v_coef * val_loss + options.e_coef * entropy_loss
                loss = clip_loss + options.v_coef * val_loss
            else:
                raise Exception(f"Method {options.method} not supported!")
            loss[loss_mask] = 0
            loss.mean().backward()
            optimizer.step()

            total_reward += rewards.detach().cpu().numpy().sum()
            total_loss += loss.detach().cpu().numpy().sum()

        val_reward = 0
        val_reward_est_0 = 0
        val_reward_est_f = 0
        for batch in tqdm(val_loader):
            # Calculate rewards for batch
            rewards = get_rewards(batch, options).to(device)
            val_reward += rewards.detach().cpu().numpy().sum()

            # Get value function estimates from retriever
            with torch.no_grad():
                _, value_estimates = retriever(**batch)
            value_estimates = value_estimates.view(batch["example_encodings"].shape[0], -1).detach().cpu().numpy()
            val_reward_est_0 += value_estimates[:, 0].sum()
            val_reward_est_f += value_estimates[:, -1].sum()

        # Report stats on current epoch
        avg_loss = total_loss / len(dataset)
        avg_reward = total_reward / len(dataset)
        avg_val_reward = val_reward / len(val_set)
        avg_val_reward_est_0 = val_reward_est_0 / len(val_set)
        avg_val_reward_est_f = val_reward_est_f / len(val_set)
        if run:
            run.log({"loss": avg_loss, "reward": avg_reward, "val_reward": avg_val_reward})
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}, Val Reward: {avg_val_reward:.4f}, "
              f"Val Reward Est (S_0): {avg_val_reward_est_0:.4f}, Val Reward Est (S_T): {avg_val_reward_est_f:.4f}")

        # Save model with best reward on validation set
        if best_reward is None or avg_val_reward >= best_reward:
            best_reward = avg_val_reward
            best_model.load_state_dict(retriever.state_dict())
            print("Best!")

    torch.save(best_model.state_dict(), f"{options.model_name}.pt")
    evaluate(run, best_model, "dev1k", options)
