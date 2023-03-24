import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import nltk

from models.retriever import retriever_model
from models.generator import Generator
from data_loading.data_types import GetDataFunction, ProcessDataFunction, CheckCorrectFunction
from data_loading.reticl_dataset import RetICLDataset, Collator, CollatedBatch
from evaluate import evaluate_reticl
from constants import SamplingMethod, RLAlgorithm, Reward
from utils import TrainOptions, device, is_pg

def get_predictions(batch: CollatedBatch, check_correct: CheckCorrectFunction):
    # Generate predictions given retrieved context and check correctness
    predictions = Generator.generate(**batch)
    correct = torch.Tensor([
        1 if check_correct(src_meta_data, pred) else 0
        for src_meta_data, pred in zip(batch["meta_data"], predictions)
    ])
    return predictions, correct

def get_rewards(batch: CollatedBatch, check_correct: CheckCorrectFunction, options: TrainOptions):
    if options.reward in (Reward.EXACT.value, Reward.EXACT_AND_BLEU.value):
        predictions, correct = get_predictions(batch, check_correct)

        if options.reward == Reward.EXACT.value:
            # Reward is 1 if prediction is correct, -1 otherwise
            return 2 * correct - 1
            # return correct

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
        nlls = Generator.get_nll(**batch)
        # return torch.exp(nlls)
        return 2 * torch.exp(-nlls) - 1

def get_td_error(value_estimates: torch.Tensor, rewards: torch.Tensor):
    batch_size = rewards.shape[0]
    # Append 0 to value estimates for terminal state
    v_t = F.pad(value_estimates.detach().view(batch_size, -1), (0, 1))
    # TD error: r_t + v_(t+1) - v_t
    return (rewards + v_t[:, 1:] - v_t[:, :-1]).view(-1)

def get_gae(value_estimates: torch.Tensor, rewards: torch.Tensor):
    lam = 0.9
    batch_size = rewards.shape[0]
    # GAE: sum_{i=t}^{T} (r_t + v_(i+1) - v_i) * lam^(T-t)
    gae = get_td_error(value_estimates, rewards).view(batch_size, -1)
    for t in range(gae.shape[1] - 2, -1, -1):
        gae[:, t] += lam * gae[:, t + 1]
    return gae.view(-1)

def get_entropy(activations: torch.Tensor):
    # H_t = -sum(pi(s,.) * log(pi(s,.)))
    # Take average over batch and all time steps
    action_distro = torch.softmax(activations, dim=-1).clip(1e-35)
    entropy = -torch.sum(action_distro * torch.log(action_distro), dim=-1).mean()
    # Normalize by maximum entropy so coefficient is independent of action space size
    return entropy / torch.log(torch.tensor(action_distro.shape[-1]))

def train_reticl(get_data: GetDataFunction, process_sample: ProcessDataFunction, check_correct: CheckCorrectFunction,
                 train_split: str, dev_split: str, options_dict: dict):
    options = TrainOptions(options_dict)
    if options.wandb:
        run = wandb.init(project="reticl", config=options.as_dict())
    else:
        run = None
    retriever = retriever_model(options)
    retriever.train()
    if options.sep_val_model:
        val_est_model = retriever_model(options)
        val_est_model.train()
    else:
        val_est_model = None
    dataset = RetICLDataset(get_data, process_sample, train_split, retriever, options)
    val_set = RetICLDataset(get_data, process_sample, dev_split, retriever, options)
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
    params = list(retriever.parameters()) + list(val_est_model.parameters()) if options.sep_val_model else list(retriever.parameters())
    optimizer = torch.optim.AdamW(params, lr=options.lr, weight_decay=options.wd)
    torch.autograd.set_detect_anomaly(True)
    print("Training...")
    previous_model = None # For PPO
    best_model = retriever_model(options)
    best_reward = None
    e_coef = 0.0
    for epoch in range(options.epochs):
        total_reward = 0
        total_loss = 0
        total_vf_loss = 0
        train_example_set = set()
        val_example_set = set()

        # Update exploration parameters
        if options.sm == SamplingMethod.EPSILON_GREEDY.value:
            dataset.update_epsilon(options.epsilon * options.expl_decay_rate ** epoch)
            print("Epsilon:", dataset.epsilon)
        elif options.rl_algo == RLAlgorithm.PPO.value:
            # e_coef = options.e_coef * options.expl_decay_rate ** epoch
            e_coef = options.e_coef * max(1 - (1 - options.expl_decay_rate) * epoch / options.epochs, 0)
            print("Entropy Coefficient:", e_coef)

        # Sample batch from dataset - example retrieval is also done here (__getitem__ in RetICLDataset)
        for batch in tqdm(data_loader):
            batch_size, max_num_examples = batch["example_encodings"].shape[:2]
            optimizer.zero_grad()

            # Calculate rewards for batch - only applied to final example in sequence
            rewards = get_rewards(batch, check_correct, options).to(device) # (N)
            rewards = F.pad(rewards.unsqueeze(1), (max_num_examples - 1, 0)) # (N x L)

            # Keep track of used examples
            for sample_idx in range(batch_size):
                for example_num, example_idx in enumerate(batch["policy_example_indices"][sample_idx]):
                    # Add penalty for repeated examples
                    # if example_idx.item() in train_example_set:
                    #     penalty = (1 / max_num_examples) * (len(dataset) / len(dataset.corpus)) * .5
                    #     rewards[sample_idx, example_num] -= penalty
                    train_example_set.add(example_idx.item())

            # Calculate returns with reverse cumulative sum, assume gamma=1
            returns = torch.cumsum(rewards.flip(1), dim=1).flip(1).view(-1) # (N * L)

            # Get activations on current examples and value function estimates from retriever
            if options.sep_val_model:
                activations, _ = retriever(**batch)
                _, value_estimates = val_est_model(**batch)
            else:
                activations, value_estimates = retriever(**batch)

            # Calculate loss and backprop
            loss_mask = torch.arange(max_num_examples).expand(batch_size, -1) >= batch["num_examples"].unsqueeze(1)
            loss_mask = loss_mask.view(-1)
            vf_loss = None
            if options.rl_algo == RLAlgorithm.MCC.value:
                loss = torch.nn.MSELoss(reduction="none")(activations, returns)
            elif options.rl_algo == RLAlgorithm.REINFORCE.value:
                # REINFORCE: param = param + lr * G * grad(log(pi[a]))
                # GD: param = param - lr * grad(loss)
                # loss = -G * log(pi[a])
                # pi[a] = softmax(activations)[a]
                # CEL = -log(softmax(activations)[a])
                # loss = G * CEL
                loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fn(activations, batch["policy_example_indices"].view(-1))
                loss = loss * returns.view(-1)
            elif options.rl_algo == RLAlgorithm.RWB.value:
                pg_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                pg_loss = pg_loss_fn(activations, batch["policy_example_indices"].view(-1))
                pg_loss = pg_loss * (returns.view(-1) - value_estimates.detach()) # Don't differentiate w.r.t. baseline
                vf_loss = torch.nn.MSELoss(reduction="none")(value_estimates, returns)
                loss = pg_loss + options.v_coef * vf_loss
            elif options.rl_algo == RLAlgorithm.AC.value:
                td_error = get_td_error(value_estimates, rewards)
                pg_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                pg_loss = pg_loss_fn(activations, batch["policy_example_indices"].view(-1))
                pg_loss = pg_loss * td_error
                # (r_t + v_(t+1) - v_t)^2 = ((r_t + v_(t+1) - v_t + v_t) - v_t)^2 = ((td_err + v_t) - v_t)^2
                vf_loss = torch.nn.MSELoss(reduction="none")(value_estimates, td_error + value_estimates.detach())
                loss = pg_loss + options.v_coef * vf_loss
            elif options.rl_algo == RLAlgorithm.PPO.value:
                # Get policy ratio
                if not previous_model:
                    ratio = torch.ones((batch_size * max_num_examples)).to(device)
                    previous_model = retriever_model(options)
                else:
                    with torch.no_grad():
                        pi_old_activations, _ = previous_model(**batch)
                    cur_policy_probs = torch.softmax(activations, dim=-1)[torch.arange(batch_size * max_num_examples), batch["policy_example_indices"].view(-1)]
                    old_policy_probs = torch.softmax(pi_old_activations, dim=-1)[torch.arange(batch_size * max_num_examples), batch["policy_example_indices"].view(-1)]
                    ratio = cur_policy_probs / old_policy_probs
                # Copy model for next iteration
                previous_model.load_state_dict(retriever.state_dict())

                # Get estimated advantage
                advantage = get_gae(value_estimates, rewards)

                # Get clip loss
                epsilon = 0.1
                clip_loss = -torch.min(ratio * advantage, torch.clip(ratio, 1 - epsilon, 1 + epsilon) * advantage)

                # Get value function loss
                vf_loss = torch.nn.MSELoss(reduction="none")(value_estimates, returns)

                # Maximize entropy - encourages exploration by flattening action distribution
                entropy = get_entropy(activations)

                # Get final loss
                # TODO: should add supplemental losses after taking mean?
                if options.sep_val_model:
                    loss = clip_loss - e_coef * entropy
                    vf_loss.mean().backward()
                else:
                    loss = clip_loss + options.v_coef * vf_loss - e_coef * entropy
            else:
                raise Exception(f"Algorithm {options.rl_algo} not supported!")
            loss[loss_mask] = 0
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(params, options.grad_clip)
            optimizer.step()

            total_reward += rewards.detach().cpu().numpy().sum()
            total_loss += loss.detach().cpu().numpy().sum()
            if vf_loss is not None:
                total_vf_loss += vf_loss.detach().cpu().numpy().sum()

        # Get average reward on validation set
        val_reward = 0
        val_entropy = None
        for batch in tqdm(val_loader):
            for example_idx in batch["policy_example_indices"].view(-1):
                val_example_set.add(example_idx.item())

            rewards = get_rewards(batch, check_correct, options).to(device)
            val_reward += rewards.detach().cpu().numpy().sum()

            if is_pg(options):
                with torch.no_grad():
                    activations, _ = retriever(**batch)
                val_entropy = get_entropy(activations)

        # Report stats on current epoch
        avg_loss = total_loss / len(dataset)
        avg_reward = total_reward / len(dataset)
        avg_val_reward = val_reward / len(val_set)
        avg_vf_loss = total_vf_loss / (len(dataset) * max_num_examples) if vf_loss is not None else None
        if run:
            run.log({
                "loss": avg_loss,
                "vf_loss": avg_vf_loss,
                "reward": avg_reward,
                "val_reward": avg_val_reward,
                "train_examples": len(train_example_set),
                "val_examples": len(val_example_set),
                "val_entropy": val_entropy,
                "epsilon": dataset.epsilon if options.sm == SamplingMethod.EPSILON_GREEDY.value else None,
            })
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}, Val Reward: {avg_val_reward:.4f}, "
              f"Train Examples: {len(train_example_set)}, Val Examples: {len(val_example_set)}")

        # Save model with best reward on validation set
        if best_reward is None or avg_val_reward > best_reward:
            best_reward = avg_val_reward
            print("Best!")
            best_model.load_state_dict(retriever.state_dict())

    # Save and evaluate final model
    final_model = best_model if options.save_best else retriever
    torch.save(final_model.state_dict(), f"{options.model_name}.pt")
    if options.reward != Reward.PPL.value:
        evaluate_reticl(run, get_data, process_sample, check_correct, final_model, dev_split, options)
