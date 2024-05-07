#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List

from gpt import make_gpt
from tokenizer import Tokenizer


def all_possible(n, k):
    # return all possible lists of k elements, each in range of [0,n)
    if k == 0:
        yield []
    else:
        for i in range(n):
            for c in all_possible(n, k - 1):
                yield [i] + c


def greedy_markov_chain(matrix):
    greedy_matrix = np.zeros_like(matrix)
    for col in range(matrix.shape[1]):
        max_idx = np.argmax(matrix[:, col])
        greedy_matrix[max_idx, col] = 1
    return greedy_matrix


def markov_chain_stable_state(matrix):
    # define your Markov matrix, the sum of every colum should be 1
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Find the index of the eigenvalue equal to 1 (the stationary eigenvalue)
    stable_index = np.where(np.isclose(eigenvalues, 1.0))[0][0]

    # The corresponding eigenvector at stable_index is the stationary distribution
    stable_state = eigenvectors[:, stable_index]

    # Normalize the stable state vector to make sure it sums to 1
    stable_state /= stable_state.sum()
    return stable_state.astype(np.float32)


def markov_matrix(gpt):
    vocab_size, context_length = gpt.config.vocab_size, gpt.config.block_size
    num_states = vocab_size ** context_length
    markov_matrix = np.zeros((num_states, num_states))
    states = list(all_possible(vocab_size, context_length))
    for state_idx, xi in enumerate(states):
        print(f"state {state_idx}: {xi}")

    for state_idx, xi in enumerate(states):
        x = torch.tensor(xi, dtype=torch.long)[None, ...]  # turn the list into a torch tensor and add a batch dimension
        logits = gpt(x)
        probs = nn.functional.softmax(logits, dim=-1)  # get the probabilities
        y = probs[0].tolist()  # remove the batch dimension and unpack the tensor into simple list
        print(f"input {xi} ---> {y}")
        for next_token, next_token_prob in enumerate(y):
            next_state = xi[1:] + [next_token]  # crop the context and append the next character
            next_idx = states.index(next_state)
            markov_matrix[next_idx, state_idx] = next_token_prob

    return markov_matrix


def logits_repeat_penalty(logits, prev_tokens: List, repeat_penalty: float = 1.0):
    tokens = torch.Tensor(list(set(prev_tokens))).long()
    score = logits[:, tokens]
    score = torch.where(score > 0, score / repeat_penalty, score * repeat_penalty)
    logits[:, tokens] = score
    return logits


def generate(
    model, input_str: str, max_length: int = 32,
    context_length: int = 2, repeat_penalty: float = 1.0,
    en: bool = False,
) -> str:
    tokenizer = Tokenizer(en=en)
    tokens = tokenizer.encode(input_str, eos=False)
    begin_length = len(tokens)
    while max_length > 0:
        input_tokens = torch.tensor(tokens[-context_length:], dtype=torch.long)[None, ...]
        logits = model(input_tokens)
        logits = logits_repeat_penalty(logits, tokens[begin_length:], repeat_penalty)
        probs = nn.functional.softmax(logits, dim=-1)
        next_token_id: int = torch.argmax(probs).tolist()
        # print(f"predict: {next_token_id}")
        tokens.append(next_token_id)
        if next_token_id == tokenizer.vocab["</s>"]:
            break
        max_length -= 1

    if tokens[-1] != tokenizer.vocab["</s>"]:
        tokens.append(tokenizer.vocab["</s>"])
    result = tokenizer.decode(tokens)
    return result


def prepare_data(data, context_length=2):
    seq = list(map(int, data))
    X, Y = [], []
    # iterate over the sequence and grab every consecutive 3 bits
    # the correct label for what's next is the next bit at each position
    for i in range(len(seq) - context_length):
        X.append(seq[i:i+context_length])
        Y.append(seq[i+context_length])
        print(f"example {i+1:2d}: {X[-1]} --> {Y[-1]}")
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    print(X.shape, Y.shape)
    return X, Y


def train(loops: int = 50, en: bool = False):
    torch.manual_seed(42)

    tokenizer = Tokenizer(en=en)
    vocab_size, context_length = len(tokenizer.vocab), 2

    train_data = "I love water I love water mark" if en else "我爱水我爱水果"

    data = "".join(str(x) for x in tokenizer.encode(train_data))
    print(data)
    X, Y = prepare_data(data, context_length)

    model, optimizer = make_gpt(vocab_size, context_length)
    for i in range(loops):
        logits = model(X)
        loss = F.cross_entropy(logits, Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"iter {i}: loss {loss.item(): .4f}")
    return model


def print_markov_matrix(model):
    np.set_printoptions(precision=3, suppress=True)
    matrix = markov_matrix(model)
    stable_state = markov_chain_stable_state(matrix).astype(np.float32)
    print(f"markov matrix:\n{matrix}")
    print(f"stable state:\n{stable_state}")


def main(repeat_penalty: float = 1.0, en: bool = False):
    """See how repeat penalty works.

    Args:
        repeat_penalty (float): The penalty factor for repeated tokens.
        en (bool): Whether to use English or Chinese train data.
    """
    model = train(loops=50, en=en)
    prefix_text = "I" if en else "我"
    gen_content = generate(
        model, prefix_text,
        max_length=64,
        repeat_penalty=repeat_penalty,
        en=en,
    )
    print(f"Generate content: {gen_content}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
