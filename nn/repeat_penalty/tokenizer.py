#!/usr/bin/env python3

from typing import List, Dict, Optional


class Tokenizer:

    def __init__(self, vocab: Optional[Dict] = None, en: bool = False):
        if vocab is None:
            if en:
                vocab = {
                    "<s>": 0,  # BOS
                    "</s>": 1,  # EOS
                    "I": 2,
                    "love": 3,
                    "water": 4,
                    "mark": 5,
                }
            else:
                vocab = {
                    "<s>": 0,  # BOS
                    "</s>": 1,  # EOS
                    "我": 2,
                    "爱": 3,
                    "水": 4,
                    "果": 5,
                    # "，": 6,
                    # "。": 7,
                }
        self.vocab = vocab
        self.en = en
        self.reverse_vocab = {v: k for k, v in vocab.items()}

    def decode(self, tokens: List[int]) -> str:
        assert tokens[0] == self.vocab["<s>"] and tokens[-1] == self.vocab["</s>"]
        link_str = " " if self.en else ""
        text = link_str.join([self.reverse_vocab[token] for token in tokens[1:-1]])
        return text

    def encode(self, text: str, eos=True) -> List[int]:
        tokens = [self.vocab["<s>"]]
        if self.en:
            text = text.split()  # split by space for English
            if not isinstance(text, list):  # text is a single word
                text = [text]

        for s in text:
            assert s in self.vocab, f"out of vocab: {s}"
            tokens.append(self.vocab[s])

        if eos:
            tokens.append(self.vocab["</s>"])
        return tokens
