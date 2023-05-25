import re
import os
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from models.reticl_rnn import RetICLRNN
from data_loading.data_types import GetDataFunction, ProcessDataFunction
from data_loading.reticl_dataset import RetICLDataset, Collator
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

def get_latent_states(retriever: RetICLRNN, dataset: RetICLDataset, options: TrainOptions):
    all_latent_states = []
    data_loader = DataLoader(
        dataset,
        collate_fn=Collator(),
        batch_size=options.batch_size,
        shuffle=False
    )
    for batch in tqdm(data_loader):
        # TODO: collect correctness?
        with torch.no_grad():
            latent_states = retriever.get_latent_states(**batch)
        all_latent_states.append(latent_states.detach().cpu().numpy())
    all_latent_states = np.concatenate(all_latent_states, axis=0)
    return all_latent_states[:, 0]

def visualize_representations(get_data: GetDataFunction, process_sample: ProcessDataFunction, options_dict: dict):
    options = TrainOptions(options_dict)
    mode = "encodings"

    # Load model
    if not options.model_name:
        retriever = None
    else:
        retriever = RetICLRNN(options).to(device)
        retriever.load_state_dict(torch.load(f"{options.model_name}.pt", map_location=device))
        retriever.eval()

    # Load data
    cache_filename = f"reprs_{options.dataset}_{str(options.model_name).replace('/', '-')}_vcs{options.val_corpus_size}.pt"
    if os.path.exists(cache_filename):
        cached_encoding_matrix = torch.load(cache_filename)
    else:
        cached_encoding_matrix = None
    dataset = RetICLDataset(get_data, process_sample, "test", retriever, options, cached_encoding_matrix=cached_encoding_matrix)
    if cached_encoding_matrix is None:
        torch.save(dataset.encoding_matrix, cache_filename)
    dataset.set_greedy(True)
    all_meta_data = [example["meta_data"] for example in dataset.corpus]

    # Group problem classes for visualization
    # all_problem_classes = [get_problem_class(example["meta_data"]["answer"]) for example in dataset.corpus]
    # problem_class_counter = Counter(all_problem_classes)
    # print(problem_class_counter.most_common())
    # classes_to_keep = 6
    # class_map = {pc: idx for idx, (pc, _) in enumerate(problem_class_counter.most_common(classes_to_keep))}
    # cmap = [class_map.get(problem_class, classes_to_keep) for problem_class in all_problem_classes]

    # age = [238,129,296,61,116,698]
    # dist = [148,406,453,80,397]
    # weight = [7,84,286]
    # money = [353,100,540,114,161,91,81,597,375]
    # cmap = []
    # for idx in range(len(dataset.corpus)):
    #     if idx in age:
    #         cmap.append(1)
    #     elif idx in dist:
    #         cmap.append(2)
    #     elif idx in weight:
    #         cmap.append(3)
    #     elif idx in money:
    #         cmap.append(4)
    #     else:
    #         cmap.append(0)

    # Number of steps
    if options.dataset == Datasets.TABMWP.value:
        cmap = [ex["encoder_label"].count("\\n") for ex in dataset.corpus]
    else:
        cmap = [ex["encoder_label"].count("\n") for ex in dataset.corpus]

    # Number of unique operations
    # cmap = []
    # for ex in dataset.corpus:
    #     num_ops = 0
    #     if re.findall(r"<<[\d\.]+\+[\d\.]+=[\d\.]+>>", ex["meta_data"]["answer"]):
    #         num_ops += 1
    #     if re.findall(r"<<[\d\.]+-[\d\.]+=[\d\.]+>>", ex["meta_data"]["answer"]):
    #         num_ops += 1
    #     if re.findall(r"<<[\d\.]+\*[\d\.]+=[\d\.]+>>", ex["meta_data"]["answer"]):
    #         num_ops += 1
    #     if re.findall(r"<<[\d\.]+/[\d\.]+=[\d\.]+>>", ex["meta_data"]["answer"]):
    #         num_ops += 1
    #     cmap.append(num_ops)

    # Get representations to visualize
    if mode == "encodings":
        reprs = dataset.encoding_matrix
    elif mode == "encodings_transformed":
        with torch.no_grad():
            reprs = torch.matmul(dataset.encoding_matrix, retriever.bilinear.T)
    elif mode == "latent_states":
        reprs = get_latent_states(retriever, dataset, options)

    # Reduce to 2 dimensions via T-SNE
    print(reprs.shape)
    tsne = TSNE(n_components=2, perplexity=30, early_exaggeration=12)
    reprs_reduced = tsne.fit_transform(reprs.detach().cpu().numpy())

    # Show representations in scatter plots
    plt.scatter(
        reprs_reduced[:, 0],
        reprs_reduced[:, 1],
        cmap="RdYlGn",
        c=cmap,
        picker=True
    )
    def onpick(event):
        ind = event.ind[0]
        print(dataset.corpus[ind]["encoder_context"] + dataset.corpus[ind]["encoder_label"])
        print(ind)
    plt.connect("pick_event", onpick)
    plt.xticks([])
    plt.yticks([])
    plt.show()
