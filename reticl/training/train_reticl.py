from typing import List
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import nltk

from reticl.models.retriever import retriever_model, Retriever
from reticl.models.generator import Generator
from reticl.data_loading.data_types import DatasetConfig
from reticl.data_loading.reticl_dataset import RetICLDataset, Collator, CollatedBatch
from reticl.training.replay_buffer import ReplayBuffer
from reticl.evaluate import evaluate_reticl
from reticl.constants import SamplingMethod, RLAlgorithm, Reward, LRSchedule
from reticl.utils import TrainOptions, device, is_pg

def get_predictions(batch: CollatedBatch, dataset_config: DatasetConfig):
    # Generate predictions given retrieved context and check correctness
    predictions = Generator.generate(**batch)
    if dataset_config.get("check_correct_batch"):
        correct = dataset_config["check_correct_batch"](
            batch["meta_data"], [pred["text"] for pred in predictions])
    else:
        correct = torch.Tensor([
            dataset_config["check_correct"](src_meta_data, pred["text"])
            for src_meta_data, pred in zip(batch["meta_data"], predictions)
        ])
    return predictions, correct

def get_rewards(batch: CollatedBatch, dataset_config: DatasetConfig, options: TrainOptions):
    if options.reward in (Reward.EXACT.value, Reward.EXACT_AND_PPL.value, Reward.EXACT_AND_BLEU.value):
        predictions, correct = get_predictions(batch, dataset_config)

        if options.reward == Reward.EXACT.value:
            # Reward is 1 if prediction is correct, -1 otherwise
            return 2 * correct - 1

        if options.reward == Reward.EXACT_AND_PPL.value:
            ppl_weight = .5
            ppl = torch.tensor([-pred["nll"] for pred in predictions]).exp()
            return 2 * (correct * (1 - ppl_weight) + ppl * ppl_weight) - 1

        if options.reward == Reward.EXACT_AND_BLEU.value:
            # Calculate bleu score on the generated solutions
            bleu = torch.Tensor([
                nltk.translate.bleu([target], pred["text"])
                for pred, target in zip (predictions, batch["labels"])
            ])
            # Half of reward comes from bleu and other half from final correctness
            return correct + bleu - 1

    # Reward is inverse perplexity assigned to label given the context
    if options.reward == Reward.PPL.value:
        nlls = Generator.get_nll(**batch)
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

def get_optim(models: List[Retriever], options: TrainOptions, checkpoint = None):
    all_named_params = []
    for model in models:
        all_named_params += list(model.named_parameters())
    retriever_params = [param for name, param in all_named_params if "encoder" not in name]
    encoder_params = [param for name, param in all_named_params if "encoder" in name]
    optimizer = torch.optim.AdamW([
        {"params": retriever_params},
        {"params": encoder_params, "lr": options.encoder_lr}
    ], lr=options.lr, weight_decay=options.wd)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint)
    if options.lr_sched == LRSchedule.LINEAR.value:
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer, start_factor=1.0, end_factor=0.0, total_iters=options.epochs)
    elif options.lr_sched == LRSchedule.CYCLE.value:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=[group["lr"] for group in optimizer.param_groups], total_steps=options.epochs, pct_start=0.1, anneal_strategy="linear"
        )
    else:
        lr_scheduler = None
    return optimizer, lr_scheduler, retriever_params + encoder_params

def polyak_update(source: torch.nn.Module, target: torch.nn.Module, tau: float):
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

def train_reticl(dataset_config: DatasetConfig, train_split: str, dev_split: str, options_dict: dict):
    options = TrainOptions(options_dict)
    if options.wandb:
        run = wandb.init(project="reticl", config=options.as_dict())
    else:
        run = None

    # Load checkpoint
    checkpoint_path = f"{options.model_name}_ckpt.pt"
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = None

    # Create/load model(s) and optimizer(s)
    if options.rl_algo == RLAlgorithm.DSAC.value and not options.sep_val_model:
        retriever = retriever_model(options, num_critics=2)
        best_model = retriever_model(options, num_critics=2)
    else:
        retriever = retriever_model(options)
        best_model = retriever_model(options)
    if checkpoint:
        retriever.load_state_dict(checkpoint["retriever"])
    retriever.train()
    if options.sep_val_model and options.rl_algo != RLAlgorithm.DSAC.value:
        val_est_model = retriever_model(options)
        if checkpoint:
            val_est_model.load_state_dict(checkpoint["val_est"])
        val_est_model.train()
    else:
        val_est_model = None
    optimizer, scheduler, retriever_params = get_optim(
        [retriever, val_est_model] if val_est_model is not None else [retriever],
        options, checkpoint["optimizer"] if checkpoint else None
    )

    # Initialize critics/targets/alpha/optimizers for DSAC
    if options.rl_algo == RLAlgorithm.DSAC.value:
        num_critics = 2
        if options.sep_val_model:
            critics = [retriever_model(options, True, False) for _ in range(num_critics)]
            critic_targets = [retriever_model(options, True, False) for _ in range(num_critics)]
            cosps = [get_optim([critic], options) for critic in critics]
            critic_optims = [cosp[0] for cosp in cosps]
            critic_schedulers = [cosp[1] for cosp in cosps]
            for critic, target in zip(critics, critic_targets):
                target.load_state_dict(critic.state_dict())
                target.eval()
        else:
            cosps = [get_optim([retriever], options) for _ in range(num_critics)]
            critic_optims = [cosp[0] for cosp in cosps]
            critic_schedulers = [cosp[1] for cosp in cosps]
            target = retriever_model(options, num_critics=2)
            target.load_state_dict(retriever.state_dict())
            target.eval()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.AdamW([log_alpha], lr=options.lr, weight_decay=options.wd)
        target_entropy = options.e_coef * torch.log(torch.tensor(options.corpus_size))

    # Create train/val datasets/loaders
    dataset = RetICLDataset(dataset_config, train_split, retriever, options)
    val_set = RetICLDataset(dataset_config, dev_split, retriever, options, False)
    val_set.set_greedy(True) # Use greedy retrieval for validation
    data_loader = DataLoader(
        dataset,
        # Actual collate done outside loader so it's easier to collect samples for adding to replay buffer
        collate_fn=lambda batch: batch,
        batch_size=options.batch_size,
        shuffle=True,
        drop_last=False
    )
    collator = Collator()
    val_loader = DataLoader(
        val_set,
        collate_fn=Collator(),
        batch_size=options.batch_size,
        shuffle=False,
        drop_last=False
    )

    print("Training...")
    # torch.autograd.set_detect_anomaly(True)
    if options.rl_algo == RLAlgorithm.PPO.value:
        previous_model = None
    if options.rl_algo == RLAlgorithm.DSAC.value:
        replay_buffer = ReplayBuffer(options)
    best_val_accuracy = None
    e_coef = 0.0
    if checkpoint:
        torch.random.set_rng_state(checkpoint["rng_state"].cpu())
    starting_epoch = 0 if checkpoint is None else checkpoint["epoch"]
    for epoch in range(starting_epoch, options.epochs):
        total_reward = 0
        total_loss = 0
        total_vf_loss = 0
        train_example_set = set()
        val_example_set = set()

        # Update exploration parameters
        cur_lr = optimizer.param_groups[0]["lr"]
        print("LR:", cur_lr)
        if options.sm == SamplingMethod.EPSILON_GREEDY.value:
            dataset.update_epsilon(options.eg_eps * options.expl_decay_rate ** epoch)
            print("Epsilon:", dataset.epsilon)
        else:
            e_coef = options.e_coef * max(1 - (1 - options.expl_decay_rate) * epoch / options.epochs, 0)
            print("Entropy Coefficient:", e_coef)

        # Sample batch from dataset - example retrieval is also done here (__getitem__ in RetICLDataset)
        for raw_batch in tqdm(data_loader):
            batch = collator(raw_batch)
            batch_size, max_num_examples = batch["example_encodings"].shape[:2]

            # Calculate rewards for batch - only applied to final example in sequence
            rewards = get_rewards(batch, dataset_config, options).to(device) # (N)
            rewards = F.pad(rewards.unsqueeze(1), (max_num_examples - 1, 0)) # (N x L)
            total_reward += rewards.detach().cpu().numpy().sum()

            # Keep track of used examples
            for example_idx in batch["policy_example_indices"].view(-1):
                train_example_set.add(example_idx.item())

            # Calculate returns with reverse cumulative sum, assume gamma=1
            returns = torch.cumsum(rewards.flip(1), dim=1).flip(1).view(-1) # (N * L)

            # Off-policy: add to buffer
            if options.rl_algo == RLAlgorithm.DSAC.value:
                # Add episodes to buffer
                replay_buffer.add(raw_batch, rewards)

                # Don't continue to training if buffer isn't full enough yet
                if len(replay_buffer) < options.episodes_before_train:
                    continue

            # On-policy: get activations on current examples and value function estimates from retriever
            if options.rl_algo != RLAlgorithm.DSAC.value:
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
                clip_loss = -torch.min(ratio * advantage, torch.clip(ratio, 1 - options.ppo_eps, 1 + options.ppo_eps) * advantage)

                # Get value function loss
                vf_loss = torch.nn.MSELoss(reduction="none")(value_estimates, returns)

                # Get final loss
                # TODO: should add supplemental losses after taking mean?
                if options.sep_val_model:
                    loss = clip_loss
                    vf_loss.mean().backward()
                else:
                    loss = clip_loss + options.v_coef * vf_loss
            elif options.rl_algo == RLAlgorithm.DSAC.value:
                # Implementation roughly based on https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
                for _ in range(options.updates_per_batch):
                    batch_size = options.train_batch_size
                    raw_batch, rewards = replay_buffer.sample(batch_size)
                    batch = collator(raw_batch)

                    # Get policy at each state
                    with torch.no_grad():
                        activations, _ = retriever(**batch)
                        pi = torch.clip(torch.softmax(activations, dim=-1), min=1e-8)
                        log_pi = torch.log(pi)

                    # Update entropy coefficient
                    alpha = torch.exp(log_alpha.detach())
                    alpha_loss = (pi * (-log_alpha * (log_pi + target_entropy))).sum(-1).mean()
                    alpha_optim.zero_grad()
                    alpha_loss.backward()
                    alpha_optim.step()

                    # Get q function targets
                    with torch.no_grad():
                        # Get q function estimates for next states, take min to avoid q value explosion
                        if options.sep_val_model:
                            next_q_ests = torch.stack([target(**batch)[0] for target in critic_targets])
                        else:
                            next_q_ests = torch.stack(target(**batch)[1])
                        next_q_ests = torch.min(next_q_ests, dim=0).values
                        soft_val_est = (pi * (next_q_ests - alpha * log_pi)).sum(dim=-1).view(batch_size, -1)
                        # Target is just reward for terminal states; and slice to start at v_1
                        soft_val_est = F.pad(soft_val_est, (0, 1))[:, 1:]
                        q_targets = rewards + soft_val_est

                    # Update critics
                    for critic_idx in range(2):
                        if options.sep_val_model:
                            q_est_vecs = critics[critic_idx](**batch)[0]
                        else:
                            q_est_vecs = retriever(**batch)[1][critic_idx]
                        # Get the estimated q value for each selected action
                        q_ests = torch.gather(
                            q_est_vecs.view(batch_size, max_num_examples, -1),
                            dim=2, index=batch["policy_example_indices"].unsqueeze(2)
                        ).squeeze(2)
                        # Minimize soft Bellman residual and update network
                        loss = F.mse_loss(q_ests, q_targets)
                        critic_optims[critic_idx].zero_grad()
                        loss.backward()
                        if options.sep_val_model:
                            torch.nn.utils.clip_grad_norm_(critics[critic_idx].parameters(), options.grad_clip)
                        else:
                            torch.nn.utils.clip_grad_norm_(retriever.parameters(), options.grad_clip)
                        critic_optims[critic_idx].step()
                        total_vf_loss += loss.item() / options.updates_per_batch

                    # Anneal targets towards critics using Polyak update
                    if options.sep_val_model:
                        for critic, target in zip(critics, critic_targets):
                            polyak_update(critic, target, options.tau)
                    else:
                        polyak_update(retriever, target, options.tau)

                    # Update actor
                    with torch.no_grad():
                        if options.sep_val_model:
                            q_ests = torch.stack([critic(**batch)[0] for critic in critics])
                        else:
                            q_ests = torch.stack(retriever(**batch)[1])
                        q_ests = torch.min(q_ests, dim=0).values
                    activations, _ = retriever(**batch)
                    pi = torch.clip(torch.softmax(activations, dim=-1), min=1e-8)
                    log_pi = torch.log(pi)
                    actor_loss = (pi * (alpha * log_pi - q_ests.detach())).sum(dim=-1).mean()
                    optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(retriever_params, options.grad_clip)
                    optimizer.step()
                    total_loss += actor_loss.item() / options.updates_per_batch
            else:
                raise Exception(f"Algorithm {options.rl_algo} not supported!")

            # Do gradient step (DSAC has its own)
            if options.rl_algo != RLAlgorithm.DSAC.value:
                # Maximize entropy - encourages exploration by flattening action distribution
                entropy = get_entropy(activations)
                loss = loss - e_coef * entropy
                loss[loss_mask] = 0
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(retriever_params, options.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.detach().cpu().numpy().sum()
                if vf_loss is not None:
                    total_vf_loss += vf_loss.detach().cpu().numpy().sum()

            # If training encoder, re-compute corpus encodings (after each training step)
            if retriever.encoder is not None:
                dataset.compute_corpus_encodings(False)

        # Update learning rate schedulers
        if scheduler is not None:
            scheduler.step()
            if options.rl_algo == RLAlgorithm.DSAC.value:
                for critic_scheduler in critic_schedulers:
                    critic_scheduler.step()

        with torch.no_grad():
            # Re-compute corpus encodings for validation set if training encoder
            if retriever.encoder is not None:
                val_set.compute_corpus_encodings()

            # Get average reward on validation set
            val_reward = 0
            val_correct = 0
            val_entropy = None
            for batch in tqdm(val_loader):
                for example_idx in batch["policy_example_indices"].view(-1):
                    val_example_set.add(example_idx.item())

                rewards = get_rewards(batch, dataset_config, options).to(device)
                val_reward += rewards.detach().cpu().numpy().sum()
                _, correct = get_predictions(batch, dataset_config)
                val_correct += correct.detach().cpu().numpy().sum()

                if is_pg(options):
                    activations, _ = retriever(**batch)
                    val_entropy = get_entropy(activations)

        # Report stats on current epoch
        avg_loss = total_loss / len(dataset)
        avg_reward = total_reward / len(dataset)
        avg_val_reward = val_reward / len(val_set)
        avg_val_accuracy = val_correct and val_correct / len(val_set)
        avg_vf_loss = total_vf_loss / (len(dataset) * max_num_examples) if total_vf_loss != 0 else None
        if run:
            run.log({
                "loss": avg_loss,
                "vf_loss": avg_vf_loss,
                "reward": avg_reward,
                "val_reward": avg_val_reward,
                "val_accuracy": avg_val_accuracy,
                "train_examples": len(train_example_set),
                "val_examples": len(val_example_set),
                "val_entropy": val_entropy,
                "epsilon": dataset.epsilon if options.sm == SamplingMethod.EPSILON_GREEDY.value else None,
                "alpha": alpha if options.rl_algo == RLAlgorithm.DSAC.value else None,
                "lr": cur_lr,
            })
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}, Val Reward: {avg_val_reward:.4f}, "
              f"Val Acc: {avg_val_accuracy:.4f}, Train Examples: {len(train_example_set)}, Val Examples: {len(val_example_set)}")

        # Save checkpoint
        # Commented since not properly resuming
        # torch.save({
        #     "retriever": retriever.state_dict(),
        #     "val_est": val_est_model.state_dict() if val_est_model is not None else None,
        #     "optimizer": optimizer.state_dict(),
        #     "rng_state": torch.random.get_rng_state(),
        #     "epoch": epoch + 1,
        # }, checkpoint_path)

        # Save model with best reward on validation set
        if best_val_accuracy is None or avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            print("Best! Saving model...")
            best_model.load_state_dict(retriever.state_dict())
            torch.save(best_model.state_dict(), f"{options.model_name}.pt")

    # Save and evaluate final model
    final_model = best_model if options.save_best else retriever
    if not options.save_best:
        torch.save(final_model.state_dict(), f"{options.model_name}.pt")
    evaluate_reticl(run, dataset_config, final_model, dev_split, options)