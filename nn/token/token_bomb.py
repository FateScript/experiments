#!/usr/bin/env python3

# reference: https://paulbutler.org/2025/smuggling-arbitrary-data-through-an-emoji/
# The key idea is to use Zero Width Joiner (ZWJ) and Variation Selector (VS) to hide the payload

from typing import List
from transformers import GPT2Tokenizer


def byte_to_variation_selector(byte: int):
    base = 0xFE00 if byte < 16 else (0xE0100 - 16)
    return chr(base + byte)


def encode(text, byte_list: List[int]) -> str:
    return text + "".join(byte_to_variation_selector(x) for x in byte_list)


def variation_selector_to_byte(vs: str) -> int:
    return ord(vs) - 0xFE00 if ord(vs) < 0xE0100 else ord(vs) - 0xE0100 + 16


def bomb_attack(hidden_text: str = "Hello, world!"):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    hidden_byte = list(hidden_text.encode("utf-8"))
    emoji = "😊"
    tokens = tokenizer.encode(emoji)
    print(f"Before hack, length of {emoji} is {len(tokens)} tokens: {tokens}")
    emoji_bomb = encode(emoji, hidden_byte)
    tokens = tokenizer.encode(emoji_bomb)
    print(f"After hack, length of {emoji_bomb} is {len(tokens)} tokens: {tokens}")


if __name__ == "__main__":
    bomb_attack()
