import re
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from models.reticl import RetICL
from data_loading.data_loading import Collator, CollatedBatch
from data_loading.tabmwp import TabMWPDataset
from data_loading.gsm8k import GSM8KDataset
from evaluate import check_correct
from constants import Datasets
from utils import TrainOptions, device

def get_problem_class(solution: str):
    return (
        ("plus", len(re.findall(r"<<[\d\.]+\+[\d\.]+=[\d\.]+>>", solution))),
        ("minus", len(re.findall(r"<<[\d\.]+-[\d\.]+=[\d\.]+>>", solution))),
        ("times", len(re.findall(r"<<[\d\.]+\*[\d\.]+=[\d\.]+>>", solution))),
        ("div", len(re.findall(r"<<[\d\.]+/[\d\.]+=[\d\.]+>>", solution))),
    )

def visualize_latent_states(options_dict: dict):
    options = TrainOptions(options_dict)

    # Load model
    retriever = RetICL(options).to(device)
    retriever.load_state_dict(torch.load(f"{options.model_name}.pt", map_location=device))
    retriever.eval()

    # Load data
    if options.dataset == Datasets.TABMWP.value:
        dataset = TabMWPDataset("train", retriever, options)
    elif options.dataset == Datasets.GSM8K.value:
        dataset = GSM8KDataset("train", retriever, options)
    else:
        raise Exception(f"Dataset {options.dataset} not supported!")
    dataset.set_greedy(True)

    # Get all examples in corpus transformed to latent space
    with torch.no_grad():
        latent_states = torch.matmul(dataset.encoding_matrix, retriever.bilinear.T)
    all_meta_data = [example["meta_data"] for example in dataset.corpus]
    all_problem_classes = [get_problem_class(example["meta_data"]["answer"]) for example in dataset.corpus]

    # Reduce to 2 dimensions via T-SNE
    tsne = TSNE(n_components=2, perplexity=40)
    latent_states_reduced = tsne.fit_transform(latent_states.detach().cpu().numpy())


    # Collect representations
    # all_latent_states = []
    # all_query_vectors = []
    # all_meta_data = []
    # all_problem_classes = []
    # data_loader = DataLoader(
    #     dataset,
    #     collate_fn=Collator(),
    #     batch_size=options.batch_size,
    #     shuffle=False
    # )
    # for batch in tqdm(data_loader):
    #     # TODO: collect correctness?
    #     with torch.no_grad():
    #         latent_states, query_vectors = retriever.get_latent_states_and_query_vectors(**batch)
    #     all_latent_states.append(latent_states.detach().cpu().numpy())
    #     all_query_vectors.append(query_vectors.detach().cpu().numpy())
    #     all_meta_data += batch["meta_data"]
    #     all_problem_classes += [get_problem_class(meta_data["answer"]) for meta_data in batch["meta_data"]]
    # all_latent_states = np.concatenate(all_latent_states, axis=0)
    # all_query_vectors = np.concatenate(all_query_vectors, axis=0)

    # # Reduce to 2 dimensions via T-SNE
    # tsne = TSNE(n_components=2, perplexity=40)
    # latent_states_reduced = [
    #     tsne.fit_transform(all_latent_states[:, idx])
    #     for idx in range(all_latent_states.shape[1])
    # ]
    # query_vectors_reduced = [
    #     tsne.fit_transform(all_query_vectors[:, idx])
    #     for idx in range(all_query_vectors.shape[1])
    # ]
    # latent_states_reduced = latent_states_reduced[0]


    # Group problem classes for visualization
    # TODO: think about other ways to define this,
    #       maybe by the number of operations in a problem or if a problem has add, minus, etc.
    problem_class_counter = Counter(all_problem_classes)
    print(problem_class_counter.most_common())
    classes_to_keep = 6
    class_map = {pc: idx for idx, (pc, _) in enumerate(problem_class_counter.most_common(classes_to_keep))}

    # Show representations in scatter plots
    plt.scatter(
        latent_states_reduced[:, 0],
        latent_states_reduced[:, 1],
        cmap="viridis",
        c=[class_map.get(problem_class, classes_to_keep) for problem_class in all_problem_classes],
        picker=True
    )
    def onpick(event):
        ind = event.ind[0]
        print(all_meta_data[ind]["question"])
        print(all_meta_data[ind]["answer"])
    plt.connect("pick_event", onpick)
    plt.show()
