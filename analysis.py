import re
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from models.reticl import RetICL
from data_loading.reticl import RetICLDataset, Collator, CollatedBatch
from data_loading.tabmwp import tabmwp_get_data, tabmwp_process_sample
from data_loading.gsm8k import gsm8k_get_data, gsm8k_process_sample
from evaluate import check_correct
from constants import Datasets
from utils import TrainOptions, device

def get_problem_class(solution: str):
    # TODO: think about other ways to define this,
    #       maybe by the total number of operations in a problem or if a problem has add, minus, etc.
    return (
        ("plus", len(re.findall(r"<<[\d\.]+\+[\d\.]+=[\d\.]+>>", solution))),
        ("minus", len(re.findall(r"<<[\d\.]+-[\d\.]+=[\d\.]+>>", solution))),
        ("times", len(re.findall(r"<<[\d\.]+\*[\d\.]+=[\d\.]+>>", solution))),
        ("div", len(re.findall(r"<<[\d\.]+/[\d\.]+=[\d\.]+>>", solution))),
    )

def get_transformed_encodings(retriever: RetICL, dataset: RetICLDataset):
    with torch.no_grad():
        return torch.matmul(dataset.encoding_matrix, retriever.bilinear.T)

def get_latent_states_and_query_vectors(retriever: RetICL, dataset: RetICLDataset, options: TrainOptions):
    all_latent_states = []
    all_query_vectors = []
    data_loader = DataLoader(
        dataset,
        collate_fn=Collator(),
        batch_size=options.batch_size,
        shuffle=False
    )
    for batch in tqdm(data_loader):
        # TODO: collect correctness?
        with torch.no_grad():
            latent_states, query_vectors = retriever.get_latent_states_and_query_vectors(**batch)
        all_latent_states.append(latent_states.detach().cpu().numpy())
        all_query_vectors.append(query_vectors.detach().cpu().numpy())
    all_latent_states = np.concatenate(all_latent_states, axis=0)
    all_query_vectors = np.concatenate(all_query_vectors, axis=0)
    return all_latent_states[:, 0], all_query_vectors[:, 0]

def visualize_representations(options_dict: dict):
    options = TrainOptions(options_dict)
    mode = "encodings"

    # Load model
    if mode == "encodings":
        retriever = None
    else:
        retriever = RetICL(options).to(device)
        retriever.load_state_dict(torch.load(f"{options.model_name}.pt", map_location=device))
        retriever.eval()

    # Load data
    if options.dataset == Datasets.TABMWP.value:
        dataset = RetICLDataset(tabmwp_get_data, tabmwp_process_sample, "train", retriever, options)
    elif options.dataset == Datasets.GSM8K.value:
        dataset = RetICLDataset(gsm8k_get_data, gsm8k_process_sample, "train", retriever, options)
    else:
        raise Exception(f"Dataset {options.dataset} not supported!")
    dataset.set_greedy(True)
    all_meta_data = [example["meta_data"] for example in dataset.corpus]

    # Group problem classes for visualization
    all_problem_classes = [get_problem_class(example["meta_data"]["answer"]) for example in dataset.corpus]
    problem_class_counter = Counter(all_problem_classes)
    print(problem_class_counter.most_common())
    classes_to_keep = 6
    class_map = {pc: idx for idx, (pc, _) in enumerate(problem_class_counter.most_common(classes_to_keep))}

    # Get representations to visualize
    if mode == "encodings":
        reprs = dataset.encoding_matrix
    elif mode == "encodings_transformed":
        with torch.no_grad():
            reprs = torch.matmul(dataset.encoding_matrix, retriever.bilinear.T)
    elif mode == "latent_states":
        reprs, _ = get_latent_states_and_query_vectors(retriever, dataset, options)

    # Reduce to 2 dimensions via T-SNE
    tsne = TSNE(n_components=2, perplexity=40)
    reprs_reduced = tsne.fit_transform(reprs.detach().cpu().numpy())

    # Show representations in scatter plots
    plt.scatter(
        reprs_reduced[:, 0],
        reprs_reduced[:, 1],
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
