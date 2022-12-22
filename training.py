import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from models.retriever import Retriever
from models.generator import Generator
from data_loading.data_loading import Collator
from data_loading.tabmwp import TabMWPDataset
from evaluate import check_correct, evaluate
from constants import SamplingMethod, Reward
from utils import TrainOptions, device

def train_retriever(options_dict: dict):
    options = TrainOptions(options_dict)
    if options.wandb:
        run = wandb.init(project="reticl", config=options.as_dict())
    else:
        run = None
    retriever = Retriever(options).to(device)
    retriever.train()
    dataset = TabMWPDataset("train", retriever, options)
    data_loader = DataLoader(
        dataset,
        collate_fn=Collator(),
        batch_size=options.batch_size,
        shuffle=True,
        drop_last=False
    )
    optimizer = torch.optim.AdamW(retriever.parameters(), lr=options.lr)
    print("Training...")
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
            if options.reward == Reward.EXACT.value:
                # Generate predictions given retrieved context
                predictions = Generator.generate(**batch)
                # Reward is 1 if prediction is correct, -1 otherwise
                rewards = torch.Tensor([
                    1 if check_correct(src_meta_data, pred) else -1
                    for src_meta_data, pred in zip(batch["meta_data"], predictions)
                ]).to(device)
            else:
                # Reward is negative perplexity assigned to label given the context
                rewards = -Generator.get_ppl(**batch)

            # Calculate returns: gamma=1 so just repeat reward over each time step
            returns = rewards.unsqueeze(1).repeat(1, max_num_examples).view(-1) # (N * L)

            # Get activations on current examples from retriever
            activations = retriever(**batch)

            # Calculate loss and backprop
            loss_mask = torch.arange(max_num_examples).expand(batch_size, -1) >= batch["num_examples"].unsqueeze(1)
            loss_mask = loss_mask.view(-1)
            if options.method == SamplingMethod.MCC.value:
                loss_fn = torch.nn.MSELoss(reduction="none")
                loss = loss_fn(activations, returns)
            else:
                # Derive GD loss function from REINFORCE update rule:
                # REINFORCE: param = param + lr * G * grad(log(pi[a]))
                # GD: param = param - lr * grad(loss)
                # loss = -G * log(pi[a])
                # pi[a] = softmax(activations)[a]
                # CEL = -log(softmax(activations)[a])
                # loss = G * CEL
                loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fn(activations, batch["policy_example_indices"].view(-1))
                loss = loss * returns.view(-1)
            loss[loss_mask] = 0
            loss.mean().backward()
            optimizer.step()

            total_reward += rewards.detach().cpu().numpy().sum()
            total_loss += loss.detach().cpu().numpy().sum()

        avg_loss = total_loss / len(dataset)
        avg_reward = total_reward / len(dataset)
        if run:
            run.log({"loss": avg_loss, "reward": avg_reward})
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}")

    torch.save(retriever.state_dict(), f"{options.model_name}.pt")
    evaluate(run, retriever, "dev1k", options)
