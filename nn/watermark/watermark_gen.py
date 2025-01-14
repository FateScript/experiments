#!/usr/bin/env python3

# reference code:
# https://github.com/huggingface/transformers/blob/b05df6611e6e3e6834acca2b50baeb7cdd5fbe3c/src/transformers/generation/logits_process.py#L2341
# watermark detector:
# https://github.com/huggingface/transformers/blob/15bd3e61f8d3680ca472c9314ad07584d20f7b81/src/transformers/generation/watermarking.py#L73

# reference paper: A Watermark for Large Language Models
# https://arxiv.org/abs/2301.10226

import math
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import List


class GPT2Watermark:

    def __init__(
        self,
        greenlist_ratio: float = 0.25,
        context: int = 2,
        hash_key: int = 15485863,  # the millionth prime
        logits_delta: float = 2.0,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.vocab_size = self.tokenizer.vocab_size

        self.greenlist_ratio = greenlist_ratio
        self.context = context
        self.hash_key = hash_key
        self.logits_delta = logits_delta

        self.do_sample = do_sample
        self.temperature = temperature

        self.rng = torch.Generator()
        self.rng.manual_seed(hash_key)
        self.seed_norm = (1 << 64) - 1

    def generate_logits(self, inputs):
        if isinstance(inputs, str):
            inputs = self.tokenizer(inputs, return_tensors="pt")
            output = self.model(**inputs)
        else:
            output = self.model(input_ids=inputs)
        logits = output.logits[..., -1, :]
        return logits

    def decode(self, input_ids) -> str:
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.flatten()
        return self.tokenizer.decode(input_ids)

    def set_seed(self, input_seq: torch.Tensor):
        # if using selfhash method, context is more usable
        context = input_seq[..., -self.context:]
        seed = self.hash_key * context[..., -1].item()
        self.rng.manual_seed(seed % self.seed_norm)

    def greenlist(self, input_seq: torch.Tensor) -> torch.Tensor:
        self.set_seed(input_seq)
        num_green_tokens = int(self.vocab_size * self.greenlist_ratio)
        green_tokens = torch.randperm(self.vocab_size, generator=self.rng)[:num_green_tokens]
        return green_tokens

    def sample_next_token(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits / self.temperature
        if self.do_sample:  # sampling decoding
            prob = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(prob, num_samples=1)
        else:  # greedy decoding
            next_token = torch.argmax(logits)
            next_token = next_token.view(1, -1)
        return next_token

    def generate(self, text: str, max_new_tokens: int = 128, watermark: bool = True):
        inputs = self.tokenizer(text, return_tensors="pt").input_ids
        for _ in range(max_new_tokens):
            logits = self.generate_logits(inputs)
            if watermark:
                greenlist = self.greenlist(inputs)
                logits[..., greenlist] += self.logits_delta  # add delta to greenlist

            next_token = self.sample_next_token(logits)
            inputs = torch.cat([inputs, next_token.view(1, -1)], dim=-1)
        return self.decode(inputs)


class WatermarkDetector:

    def __init__(
        self,
        greenlist_ratio: float = 0.25,
        context: int = 2,
        hash_key: int = 15485863,
        logits_delta: float = 2.0,
        z_threshold: float = 4.0,
    ):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.vocab_size = self.tokenizer.vocab_size

        self.greenlist_ratio = greenlist_ratio
        self.context = context
        self.hash_key = hash_key
        self.logits_delta = logits_delta
        self.z_threshold = z_threshold

        self.rng = torch.Generator()
        self.rng.manual_seed(hash_key)
        self.seed_norm = (1 << 64) - 1

    def is_greentoken(self, prefix: torch.Tensor, target_token: torch.Tensor) -> bool:
        seed = self.hash_key * prefix[..., -1].item()
        self.rng.manual_seed(seed % self.seed_norm)

        num_green_tokens = int(self.vocab_size * self.greenlist_ratio)
        green_tokens = torch.randperm(self.vocab_size, generator=self.rng)[:num_green_tokens]
        return target_token in green_tokens

    def n_gram_stats(self, input_seq) -> List[bool]:
        seq_length = input_seq.size(-1)
        indices = torch.arange(seq_length - self.context).view(-1, 1) + torch.arange(self.context + 1).view(1, -1)  # noqa
        ngrams = input_seq[..., indices].squeeze(0)

        statistics = []
        for prefix_target in ngrams:
            prefix, target = prefix_target[:-1], prefix_target[-1]
            stat = self.is_greentoken(prefix, target)
            statistics.append(stat)
        return statistics

    def compute_z_score(self, token_stats) -> float:
        green_token_count = sum(token_stats)
        expect_token_count = len(token_stats) * self.greenlist_ratio
        std = math.sqrt(expect_token_count * (1 - self.greenlist_ratio))
        return (green_token_count - expect_token_count) / std

    def detect(self, text) -> bool:
        input_seq = self.tokenizer(text, return_tensors="pt").input_ids
        token_statistics = self.n_gram_stats(input_seq)
        z_score = self.compute_z_score(token_statistics)
        return z_score >= self.z_threshold


def watermark_and_detect(text: str, seed: int = 42):
    torch.manual_seed(seed)
    watermark_model = GPT2Watermark(do_sample=True)
    detector = WatermarkDetector()

    simple_text = watermark_model.generate(text, watermark=False)
    simple_detect_result = detector.detect(simple_text)
    print(f"normal text: {simple_text}\n")
    print(f"Detection result: {simple_detect_result}\n\n")

    watermarked_text = watermark_model.generate(text)
    watermark_detect_result = detector.detect(watermarked_text)
    print(f"watermarked text: {watermarked_text}\n")
    print(f"Detection result: {watermark_detect_result}")


if __name__ == "__main__":
    watermark_and_detect("Hi, I am a watermarking model. My favorite number is")
