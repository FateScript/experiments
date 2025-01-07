#!/usr/bin/env python3

# reference code:
# https://github.com/huggingface/transformers/blob/5c75087aeee7081025370e10d1f571a11600f1ae/src/transformers/generation/utils.py#L2456
# blog post: https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38

from typing import Dict, List
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline


def generate(text: str, temperature: float = 0.01, max_new_tokens: int = 8) -> str:
    generator = pipeline('text-generation', model='gpt2', temperature=temperature)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    pad_token = tokenizer.eos_token_id

    gen_text = generator(
        text,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token
    )[0]["generated_text"]
    return gen_text


def search_vocab(prefix: str, vocab: Dict[str, int]) -> List[int]:
    # transformers/generation/utils.py#L2456 using a Trie data structure
    # for efficient search of tokens with a given prefix.
    # but for simplicity, we use a linear search here.
    token_ids = []
    for text, token in vocab.items():
        if text.startswith(prefix):
            token_ids.append(token)
    return token_ids


def token_heal_generate(text: str) -> str:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_tokens = tokenizer.encode(text)
    prefix_tokens, last_token = input_tokens[:-1], input_tokens[-1]
    last_token_text = tokenizer.convert_ids_to_tokens(last_token)

    heal_candidates = search_vocab(last_token_text, tokenizer.get_vocab())
    if len(heal_candidates) == 1:  # only one candidate, no need to heal token
        return generate(text)

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    output = model(input_ids=torch.tensor([prefix_tokens]))
    logits = output.logits[..., -1, :]
    heal_logits = logits[..., heal_candidates]
    max_prob_token = torch.argmax(heal_logits).item()
    heal_idx = heal_candidates[max_prob_token]
    healed_text = tokenizer.decode(prefix_tokens + [heal_idx])
    return generate(healed_text)


if __name__ == "__main__":
    prompts = ["Here is a link: https://", "Here is a link: https:", "Here is a link: https:/"]
    for p in prompts:
        gen_text = generate(p)
        healed_text = token_heal_generate(p)
        print(f"Original: {p}")
        print(f"Before token healing: {gen_text}")
        print(f"After token healing: {healed_text}")
        print("---" * 30)
