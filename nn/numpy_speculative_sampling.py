"""
NumPy-only speculative decoding.

The target model returns deterministic random logits for a given prefix.
The draft model reuses those logits and adds small deterministic noise, so it
is close to the target but not identical.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np


EPS = 1e-12


@dataclass
class Stats:
    draft_tokens: int = 0
    accepted_tokens: int = 0
    target_resamples: int = 0
    target_extra_samples: int = 0
    target_evals: int = 0


def _context_seed(seed: int, prefix: Iterable[int]) -> int:
    """Build a stable RNG seed from a model seed and the current token prefix."""
    tokens = np.asarray(list(prefix), dtype=np.int64)
    h = hashlib.blake2b(digest_size=8)
    h.update(int(seed).to_bytes(8, "little", signed=True))
    h.update(len(tokens).to_bytes(8, "little", signed=False))
    h.update(tokens.tobytes())
    return int.from_bytes(h.digest(), "little", signed=False)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    finite = np.isfinite(logits)
    if not np.any(finite):
        return np.full_like(logits, 1.0 / logits.size, dtype=np.float64)

    shifted = logits - np.max(logits[finite])
    exp = np.exp(np.where(finite, shifted, -np.inf))
    total = exp.sum()
    if total <= EPS:
        return np.full_like(logits, 1.0 / logits.size, dtype=np.float64)
    return exp / total


def normalize_logits(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> np.ndarray:
    """Convert logits to a probability distribution with optional filtering."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    filtered = np.asarray(logits, dtype=np.float64) / temperature
    vocab_size = filtered.size

    if 0 < top_k < vocab_size:
        keep_idx = np.argpartition(filtered, -top_k)[-top_k:]
        mask = np.full(vocab_size, -np.inf, dtype=np.float64)
        mask[keep_idx] = filtered[keep_idx]
        filtered = mask

    probs = softmax(filtered)

    if 0.0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        sorted_probs = probs[order]
        cumulative = np.cumsum(sorted_probs)
        keep_sorted = (cumulative - sorted_probs) <= top_p
        keep_sorted[0] = True

        kept = np.zeros_like(probs)
        keep_idx = order[keep_sorted]
        kept[keep_idx] = probs[keep_idx]
        total = kept.sum()
        probs = kept / total if total > EPS else probs

    return probs


def sample(probs: np.ndarray, rng: np.random.Generator) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total <= EPS:  # uniform sample
        probs = np.full(probs.size, 1.0 / probs.size, dtype=np.float64)
    else:
        probs = probs / total
    return int(rng.choice(probs.size, p=probs))


def residual_distribution(target_probs: np.ndarray, draft_probs: np.ndarray) -> np.ndarray:
    """Normalize max(target_probs - draft_probs, 0)."""
    residual = np.maximum(target_probs - draft_probs, 0.0)
    total = residual.sum()
    if total <= EPS:
        return target_probs / target_probs.sum()
    return residual / total


@dataclass(frozen=True)
class RandomTargetModel:
    vocab_size: int
    seed: int = 0
    logit_scale: float = 1.0

    def predict(self, prefix: Iterable[int]) -> np.ndarray:
        rng = np.random.default_rng(_context_seed(self.seed, prefix))
        return rng.normal(0.0, self.logit_scale, size=self.vocab_size)


@dataclass(frozen=True)
class NoisyDraftModel:
    target_model: RandomTargetModel
    seed: int = 1
    noise_scale: float = 0.35

    def predict(self, prefix: Iterable[int]) -> np.ndarray:
        base_logits = self.target_model.predict(prefix)
        rng = np.random.default_rng(_context_seed(self.seed, prefix))
        noise = rng.uniform(-self.noise_scale, self.noise_scale, size=base_logits.size)
        return base_logits + noise


def speculative_sampling(
    prefix: Iterable[int],
    draft_model: NoisyDraftModel,
    target_model: RandomTargetModel,
    max_new_tokens: int,
    gamma: int = 4,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    seed: int | None = None,
    verbose: bool = False,
) -> tuple[list[int], Stats]:
    """Generate tokens with speculative decoding.

    Args:
        prefix: Initial token ids.
        draft_model: Fast approximate model q.
        target_model: Slower target model p.
        max_new_tokens: Number of new tokens to append.
        gamma: Number of draft tokens proposed per round.
        temperature/top_k/top_p: Sampling controls applied to both models.
        seed: RNG seed for token sampling and accept/reject draws.
        verbose: Print each accept/reject decision.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if draft_model.target_model.vocab_size != target_model.vocab_size:
        raise ValueError("draft and target vocab sizes must match")

    rng = np.random.default_rng(seed)
    output = list(prefix)
    target_len = len(output) + max_new_tokens
    stats = Stats()

    while len(output) < target_len:
        draft_steps = min(gamma, target_len - len(output))
        draft_output = output.copy()
        draft_tokens = []
        draft_history_probs = []

        for _ in range(draft_steps):
            logits = draft_model.predict(draft_output)
            p = normalize_logits(logits, temperature, top_k=top_k, top_p=top_p)
            sample_token = sample(p, rng=rng)

            draft_history_probs.append(p)
            draft_output.append(sample_token)
            draft_tokens.append(sample_token)
        stats.draft_tokens += draft_steps

        target_output = output.copy()
        draft_all_accept = True
        # in real world, model verify all tokens in parallel
        for i, token in enumerate(draft_tokens):
            stats.target_evals += 1
            target_p = normalize_logits(
                target_model.predict(target_output),
                temperature, top_k=top_k, top_p=top_p
            )
            draft_p = draft_history_probs[i]
            # NOTE: reject sampling, prev write a bug here, using draft[token] / target[token]
            choose_prob = min(1.0, target_p[token] / max(draft_p[token], EPS))
            choose_token = rng.random() < choose_prob

            if choose_token:
                stats.accepted_tokens += 1
                output.append(token)
                target_output.append(token)
                if verbose:
                    print(f"accept token={token}, accept_prob={choose_prob:.4f}")
                continue
        
            # failed, resample a token
            draft_all_accept = False
            residual_p = residual_distribution(target_p, draft_p)
            token = sample(residual_p, rng=rng)
            if verbose:
                print(f"resample token={token}, prob={residual_p[token]:.4f}")
            output.append(token)
            stats.target_resamples += 1
            break  # bug: missing at first write

        # bonus token
        if draft_all_accept and (len(output) < target_len):  # NOTE: missing target_len check
            bonus_p = normalize_logits(
                target_model.predict(output),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            bouns_token = sample(bonus_p, rng=rng)
            if verbose:
                print(f"all accepted, bonus token={bouns_token}, prob={bonus_p[bouns_token]:.4f}")

            output.append(bouns_token)
            stats.target_evals += 1
            stats.target_extra_samples += 1

    return output, stats


def parse_prefix(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--prefix", type=str, default="1,2,3")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--target-seed", type=int, default=123)
    parser.add_argument("--draft-seed", type=int, default=456)
    parser.add_argument("--sample-seed", type=int, default=789)
    parser.add_argument("--logit-scale", type=float, default=1.0)
    parser.add_argument("--draft-noise", type=float, default=0.35)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be > 0")

    prefix = parse_prefix(args.prefix)
    if any(token < 0 or token >= args.vocab_size for token in prefix):
        raise ValueError("all prefix tokens must be within [0, vocab_size)")

    target = RandomTargetModel(
        vocab_size=args.vocab_size,
        seed=args.target_seed,
        logit_scale=args.logit_scale,
    )
    draft = NoisyDraftModel(
        target_model=target,
        seed=args.draft_seed,
        noise_scale=args.draft_noise,
    )

    output, stats = speculative_sampling(
        prefix=prefix,
        draft_model=draft,
        target_model=target,
        max_new_tokens=args.max_new_tokens,
        gamma=args.gamma,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.sample_seed,
        verbose=args.verbose,
    )

    print(f"prefix:    {prefix}")
    print(f"generated: {output[len(prefix):]}")
    print(f"output:    {output}")
    print(f"stats:     {stats}")


if __name__ == "__main__":
    main()
