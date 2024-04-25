#!/usr/bin/env python3

# To explore semantic in language model's embeding.
# 1. Is Embeding(man) - Embeding(woman) = Embeding(king) - Embeding(queen)?
# 2. Suppose plur vector is defined as Embeding(cats) - Embeding(cat),
# Does dot product of (plur, Embeding(one)) smaller than (plur, Embeding(two))?

import heapq
from collections import Counter
from pprint import pprint, pformat
from typing import Callable, Dict, List

import numpy as np
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


def model_and_tokenizer(repo: str = "gpt2"):
    model = AutoModelForCausalLM.from_pretrained(repo)
    tokenizer = AutoTokenizer.from_pretrained(repo)
    return model, tokenizer


def embedings_svd(weights, svd_dim: int = 2) -> np.ndarray:
    u, s, v_t = np.linalg.svd(weights, full_matrices=False)
    reduced_w = np.dot(u[:, :svd_dim], np.diag(s[:svd_dim]))
    return reduced_w


def token_to_coord(model_name: str = "gpt2", svd_dim: int = 2) -> Dict:
    model, tokenizer = model_and_tokenizer(model_name)
    ebd_weights = model.transformer.wte.weight
    np_ebd_weights = ebd_weights.detach().cpu().numpy()
    print(f"Decomposing embeding weights, svd dim: {svd_dim}...")
    reduce_ebd = embedings_svd(np_ebd_weights, svd_dim=svd_dim)

    token_coord_mapping = {}
    for token_text, token in tokenizer.get_vocab().items():
        token_coord_mapping[token_text] = reduce_ebd[token]
    return token_coord_mapping


def distance(x, y):
    return np.linalg.norm(x - y)  # Euclidean distance


def nearest_neighbors(
    text_or_vector,
    mappings: Dict,
    k: int = 5,
    distance_func: Callable = distance,
) -> Dict[str, float]:
    vector = text_or_vector
    if isinstance(text_or_vector, str):
        vector = mappings[text_or_vector]

    topk_map = heapq.nsmallest(
        k,
        ((key, distance_func(mappings[key], vector)) for key in mappings),
        key=lambda x: x[1]
    )
    return {k: v for k, v in reversed(topk_map)}


def semantic_search(from_key1: str, from_key2: str, to_key2: str):
    """
    Using svd to reduce the dimension of the embedding weights and search for the nearest neighbors.
    Using `E` to represents embeding, suppose E(father) - E(mother) = E(man) - E(woman),
    So we could search for the nearest neighbors of E(woman) - E(man) + E(father)
    and check if `mother` is in the top-k results.
    """
    model_name = "gpt2"
    logger.info(f"Start searching neighbors for {from_key1} <-> {from_key2}, (?) <-> {to_key2}...")
    topks = []
    for dim in [64, 128, 256]:
        mapping = token_to_coord(model_name, svd_dim=dim)
        from1, from2, to2 = mapping[from_key1], mapping[from_key2], mapping[to_key2]
        topk = nearest_neighbors(from1 - from2 + to2, mapping, k=5)
        topks.append(topk)
    top_freq = Counter([k for x in topks for k in x]).most_common(3)
    pprint(topks)
    max_freq = top_freq[0][1]
    search_results = []
    for key, freq in top_freq:
        if freq < max_freq:
            break
        freq_sum = sum(x[key] for x in topks)
        search_results.append((key, freq_sum))
    sorted_results = sorted(search_results, key=lambda x: x[1])
    logger.info(sorted_results)


def plur_vectors(vector_keys: List[str], dims: List[int] = [64, 128, 256]):
    model_name = "gpt2"
    for dim in dims:
        logger.info(f"Dimention: {dim}")
        mapping = token_to_coord(model_name, svd_dim=dim)
        plur = mapping["cats"] - mapping["cat"]
        values = []
        for k in vector_keys:
            align_value = plur @ mapping[k]
            values.append((k, align_value))
        sorted_values = sorted(values, key=lambda x: x[1], reverse=True)
        logger.info(f"Sorted value:\n{pformat(sorted_values)}")


def main():
    logger.info("Emedding semantic search...")
    semantic_search("dog", "cat", "cats")
    semantic_search("woman", "man", "father")
    semantic_search("1", "2", "second")

    logger.info("Plur vector align...")
    plur_vectors(["one", "two", "three", "four", "five", "six", "seven"], dims=[64, 128])
    plur_vectors([str(x) for x in range(1, 8)], dims=[64, 128])


if __name__ == "__main__":
    main()
