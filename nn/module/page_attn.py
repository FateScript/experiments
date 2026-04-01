#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field


BLOCK_SIZE = 4
NUM_BLOCKS = 16
HEAD_DIM = 8
NUM_KV_HEADS = 2


class PhysicalStorage:
    """Physical block storage, storing actual KV cache"""

    def __init__(self) -> None:
        self.memory = torch.empty(NUM_BLOCKS, BLOCK_SIZE, 2, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
        self._free_blocks: set[int] = set([i for i in range(NUM_BLOCKS)])
        self._ref_count: list[int] = [0 for _ in range(NUM_BLOCKS)]

    def free_blocks_count(self) -> int:
        return len(self._free_blocks)

    def allocate_blocks(self, block_size: int = 1) -> list[int]:
        blocks = []
        if self.free_blocks_count() < block_size:
            print("No block allocate")
            return blocks
        else:
            while len(blocks) < block_size:
                blocks.append(self._free_blocks.pop())
            return blocks

    def add_ref(self, block_idx: int):
        self._ref_count[block_idx] += 1

    def ref_count(self, block_idx: int) -> int:
        return self._ref_count[block_idx]

    def release(self, block_idx) -> bool:
        if self._ref_count[block_idx] == 0:
            print("Can't release block that already release")
            return True

        self._ref_count[block_idx] -= 1
        if self._ref_count[block_idx] == 0:
            self._free_blocks.add(block_idx)
            return True
        return False

    def write_kv(self, block_id, slot, k, v):
        self.memory[block_id, slot, 0] = k
        self.memory[block_id, slot, 1] = v

    def write_block_kv(self, block_id, k, v):
        assert k.shape == v.shape
        block_size = k.shape[0]
        self.memory[block_id, :block_size, 0] = k
        self.memory[block_id, :block_size, 1] = v

    def read_kv(self, block_id: int, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self.memory[block_id, slot]
        return k, v

    def read_block_kv(self, block_id: int):
        k, v = torch.split(self.memory[block_id], split_size_or_sections=1, dim=1)
        return k, v


MEMORY = PhysicalStorage()


@dataclass
class Sequence:
    """
    Represents an ongoing inference request.
    block_table: logical Block -> physical Block
    """
    seq_id: str
    block_table: list[int] = field(default_factory=list)
    num_tokens: int = 0     # Number of tokens written

    def allocate_sequence(self, k, v):
        seq_len = k.shape[0]
        blocks_needed = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        if blocks_needed > MEMORY.free_blocks_count():
            raise Exception("could not allocate memory")
        physical_blocks = MEMORY.allocate_blocks(blocks_needed)
        for block_id in physical_blocks:
            start = self.num_tokens
            end = min(start + BLOCK_SIZE, seq_len)
            MEMORY.write_block_kv(block_id, k[start:end, ...], v[start:end, ...])
            MEMORY.add_ref(block_id)
            self.num_tokens += (end - start)
        self.block_table.extend(physical_blocks)

    def append_token(self, k, v):
        slot_idx = self.num_tokens % BLOCK_SIZE
        if slot_idx == 0:  # new block required
            block_idx = MEMORY.allocate_blocks()
            assert len(block_idx) == 1
            self.block_table.append(block_idx[0])
            MEMORY.add_ref(block_idx[0])
        else:
            block_idx = self.block_table[-1]

        MEMORY.write_kv(block_idx, slot_idx, k, v)
        self.num_tokens += 1

    def gather_kv(self):
        k_cache, v_cache = [], []
        for block_idx in self.block_table:
            k, v = MEMORY.read_block_kv(block_idx)
            k_cache.append(k)
            v_cache.append(v)
        k, v = torch.cat(k_cache, dim=0), torch.cat(v_cache, dim=0)
        return k[:self.num_tokens].squeeze(dim=1), v[:self.num_tokens].squeeze(dim=1)

    def fork(self, fork_point: int, new_seq_id: str) -> "Sequence":
        slot_idx = fork_point % BLOCK_SIZE
        max_block = fork_point // BLOCK_SIZE
        blocks = self.block_table[:max_block]

        child_seq = Sequence(new_seq_id, block_table=blocks.copy(), num_tokens=fork_point)
        for block_id in blocks:
            MEMORY.add_ref(block_id)
        if slot_idx != 0:  # Copy-on-Write
            target_block = MEMORY.allocate_blocks(1)[0]
            src_block = self.block_table[max_block]
            for slot in range(slot_idx):
                k, v = MEMORY.read_kv(src_block, slot)
                MEMORY.write_kv(target_block, slot, k, v)

            child_seq.block_table.append(target_block)
            MEMORY.add_ref(target_block)
        return child_seq

    def free(self):
        for block in self.block_table:
            MEMORY.release(block)


class PagedAttention:

    def __init__(self, num_heads: int, head_dim: int, dtype=torch.bfloat16):
        self.num_heads = num_heads
        self.head_dim = head_dim
        hidden_size = self.num_heads * self.head_dim

        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, dtype=dtype)

    def forward_prefill(self, hidden_states: torch.Tensor, sequence: Sequence) -> torch.Tensor:
        """
        Prefill stage: process prompt and allocate blocks
        hidden_states: [seq_len, hidden_size]
        """
        seq_len, hidden_size = hidden_states.shape
        assert hidden_size == NUM_KV_HEADS * HEAD_DIM

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(seq_len, self.num_heads, self.head_dim)
        k = k.view(seq_len, self.num_heads, self.head_dim)
        v = v.view(seq_len, self.num_heads, self.head_dim)

        # allocate physical blocks for the entire sequence (Paged)
        sequence.allocate_sequence(k, v)

        scale = self.head_dim ** -0.5
        scores = torch.einsum("shd,thd->sht", q, k) * scale  # [seq_len, heads, seq_len]

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))

        attn = F.softmax(scores, dim=2)
        out = torch.einsum("sht,thd->shd", attn, v)  # [seq_len, heads, head_dim]
        out = out.reshape(seq_len, -1)

        return self.o_proj(out)

    def forward_decode(
        self,
        hidden_state: torch.Tensor,  # [1, hidden_size]
        sequence: Sequence,
    ) -> torch.Tensor:
        """
        Decode stage: incremental generation, update KV cache
        """
        # Q/K/V of new token
        q = self.q_proj(hidden_state).view(1, self.num_heads, -1)
        new_k = self.k_proj(hidden_state).view(1, self.num_heads, -1)
        new_v = self.v_proj(hidden_state).view(1, self.num_heads, -1)

        # Append to KV cache (Paged)
        sequence.append_token(new_k, new_v)

        # Gather all historical KV from physical blocks (non-contiguous memory access)
        # k_cache: [position+1, num_heads, head_dim]
        k_cache, v_cache = sequence.gather_kv()

        # Attention
        scale = self.head_dim ** -0.5
        scores = torch.einsum("qhd,khd->qhk", q, k_cache) * scale  # [1, position+1, heads]
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("qhk,khd->qhd", attn, v_cache)  # [1, heads, head_dim]
        out = out.reshape(1, -1)
        return self.o_proj(out)


@torch.no_grad()
def demo():
    # settings
    num_heads = NUM_KV_HEADS
    head_size = HEAD_DIM
    hidden_size = NUM_KV_HEADS * HEAD_DIM
    dtype = torch.bfloat16

    # initialize attention and sequence
    seq = Sequence("seq")
    attn = PagedAttention(num_heads, head_size, dtype=dtype)

    # ====== Prefill ======
    print("\n" + "=" * 60)
    print("Prefill stage (process prompt: 10 tokens)")
    print("=" * 60)

    prompt_len = 10
    hidden = torch.randn(prompt_len, hidden_size).to(dtype)

    output = attn.forward_prefill(hidden, seq)

    print(f"Input length: {seq.num_tokens}")
    print(f"Allocated physical blocks: {seq.block_table}")

    # ====== Decode ======
    print("\n" + "=" * 60)
    print("Decode stage (generate 3 tokens incrementally)")
    print("=" * 60)

    new_tokens = []
    for i in range(3):
        new_token = torch.randn(1, hidden_size).to(dtype)
        output = attn.forward_decode(new_token, seq)
        print(f"Generate token {i + 1} (position {seq.num_tokens}): using block {seq.block_table[-1]}")
        new_tokens.append(new_token)

        seq_temp = Sequence("temp")
        prefill_hidden = torch.cat([hidden, *new_tokens[:i+1]], dim=0)
        prefill_output = attn.forward_prefill(prefill_hidden, seq_temp)

        if not torch.equal(prefill_output[-1].flatten(), output.flatten()):
            print(f"Not equal at step {i}")
        seq_temp.free()

    # ====== Memory sharing (Fork/Copy-on-Write) ======
    print("\n" + "=" * 60)
    print("Fork for multiple decoding paths")
    print("=" * 60)

    child_id = "child_seq"
    fork_point = 9

    print("\nRef count:")
    for block_id in seq.block_table:
        print(f"  Block {block_id}: ref_count={MEMORY.ref_count(block_id)}")

    # fork a child sequence
    child_seq = seq.fork(fork_point, child_id)
    print("\nAfter fork, ref count:")
    for block_id in seq.block_table:
        print(f"  Block {block_id}: ref_count={MEMORY.ref_count(block_id)}")
    print(f"Child sequence: {child_seq}")
    for i in range(4):
        new_token = torch.randn(1, hidden_size).to(dtype)
        output = attn.forward_decode(new_token, child_seq)
    print(f"Child sequence after decode: {child_seq}")
    child_seq.free()


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    demo()
