#!/usr/bin/env python3

# Reimplement the MPI functions with pure python.
# Reference: https://github.com/facebookincubator/gloo
# See https://github.com/mpi4py/mpi4py if you want to use MPI in Python.

import math
import multiprocessing
import os
import time
from typing import Callable, Dict

__all__ = [
    "get_rank",
    "get_world_size",
    "get_pipes",
    "barrier",
    "broadcast",
    "gather",
    "scatter",
    "all_gather",
    "reduce",
    "all_reduce",
]


_QUEUE = None
_P2P_PIPES = None
_SYNC_COUNTER = None


def make_p2p_pipe(num_processes: int) -> Dict:
    pipe_pairs = {}
    for i in range(num_processes):
        for j in range(i + 1, num_processes):
            (src, dst) = multiprocessing.Pipe()  # bidirectional
            pipe_pairs[(i, j)] = src
            pipe_pairs[(j, i)] = dst
    return pipe_pairs


def set_pipes(pipes):
    global _P2P_PIPES
    _P2P_PIPES = pipes


def get_pipes():
    global _P2P_PIPES
    return _P2P_PIPES


def set_queue(queue):
    global _QUEUE
    _QUEUE = queue


def get_queue():
    global _QUEUE
    return _QUEUE


def set_shared_mem(counter):
    global _SYNC_COUNTER
    _SYNC_COUNTER = counter


def get_shared_mem():
    global _SYNC_COUNTER
    return _SYNC_COUNTER


def set_rank(rank):
    os.environ["RANK"] = str(rank)


def get_rank() -> int:
    return int(os.environ["RANK"])


def set_world_size(world_size):
    os.environ["WORLD_SIZE"] = str(world_size)


def get_world_size() -> int:
    return int(os.environ["WORLD_SIZE"])


def init_env(rank, world_size, queue, shared_mem, pipe_pairs):
    # similar to context in gloo
    set_rank(rank)
    set_world_size(world_size)
    set_queue(queue)
    set_shared_mem(shared_mem)
    set_pipes(pipe_pairs)


def virtual_rank(rank: int, source_rank: int, world_size: int) -> int:
    # virtual rank is the rank value if the source rank is treated as rank 0
    return (rank - source_rank) % world_size


def original_rank(virtual_rank: int, source_rank: int, world_size: int) -> int:
    return (virtual_rank + source_rank) % world_size


def barrier_shared_var():
    # using a shared variable to synchronize processes
    # NOTE: this method has `concurrency` and `reinitialization` issues
    world_size = get_world_size()
    sync_counter = get_shared_mem()

    with sync_counter.get_lock():
        sync_counter.value += 1

    while sync_counter.value < world_size:
        time.sleep(0.001)


def barrier_brooks_algorithm():
    # reference paper: https://www.inf.ed.ac.uk/teaching/courses/ppls/BarrierPaper.pdf
    rank = get_rank()
    world_size = get_world_size()
    pipes = get_pipes()

    size = 1
    while size < world_size:
        # is_direction_up = (rank % (2 * size)) < size
        # comm_rank = (rank + size) % world_size if is_direction_up else (rank - size) % world_size
        comm_rank = rank ^ size  # the same as above
        pipes[(rank, comm_rank)].send(True)  # just send a signal
        pipes[(rank, comm_rank)].recv()
        size <<= 1


def barrier():
    # dissemination barrier algorithm
    # reference in gloo:
    # https://github.com/facebookincubator/gloo/blob/81925d1c674c34f0dc34dd9a0f2151c1b6f701eb/gloo/barrier.cc#L18
    rank = get_rank()
    world_size = get_world_size()
    pipes = get_pipes()

    size = 1
    while size < world_size:
        to_rank = (rank + size) % world_size
        pipes[(rank, to_rank)].send(True)  # just send a signal
        from_rank = (rank - size) % world_size
        pipes[(rank, from_rank)].recv()
        size <<= 1


def gather_shared_queue(data, target_rank: int = 0):
    """Using a shared queue to gather data. This method is not recommended."""
    rank = get_rank()
    world_size = get_world_size()
    queue = get_queue()

    gathered_data = [None for _ in range(world_size)]
    if rank != target_rank:
        queue.put((rank, data))
        return None
    else:
        gathered_data[rank] = data

        while any([x is None for x in gathered_data]):
            from_rank, send_data = queue.get()
            gathered_data[from_rank] = send_data
        return gathered_data


def gather(data, target_rank: int = 0):
    # reference in gloo:
    # https://github.com/facebookincubator/gloo/blob/81925d1c674c34f0dc34dd9a0f2151c1b6f701eb/gloo/gather.cc#L18
    rank = get_rank()
    pipes = get_pipes()

    if rank != target_rank:
        pipes[(rank, target_rank)].send(data)
        return None
    else:
        world_size = get_world_size()
        gathered_data = []
        for src_rank in range(world_size):
            if src_rank == rank:
                gathered_data.append(data)
                continue
            recv_data = pipes[(rank, src_rank)].recv()
            gathered_data.append(recv_data)

        return gathered_data


def split_data(data, world_size: int):
    # data could be any structure that support slicing (e.g. list, tuple, numpy array)
    assert len(data) % world_size == 0, "Data length must be divisible by world_size."
    return [data[i::world_size] for i in range(world_size)]


def scatter(data, source_rank: int = 0):
    # reference in gloo:
    # https://github.com/facebookincubator/gloo/blob/81925d1c674c34f0dc34dd9a0f2151c1b6f701eb/gloo/scatter.cc#L19
    rank = get_rank()
    pipe_pairs = get_pipes()

    if rank == source_rank:
        world_size = get_world_size()
        scatter_data = split_data(data, world_size)
        for dst_rank, dst_data in enumerate(scatter_data):
            if dst_rank == source_rank:
                data = dst_data
                continue
            pipe_pairs[(source_rank, dst_rank)].send(dst_data)
        return data
    else:
        data = pipe_pairs[(rank, source_rank)].recv()
        return data


def op_to_func(op: str) -> Callable:
    assert op in ["sum", "mean", "max", "min"], f"Invalid operation: {op}."
    if op == "sum":
        return sum
    elif op == "mean":
        return lambda x: sum(x) / len(x)
    elif op == "max":
        return max
    elif op == "min":
        return min


def reduce(data, target_rank: int = 0, op: str = "sum"):
    op_func = op_to_func(op)
    rank = get_rank()
    world_size = get_world_size()
    pipes = get_pipes()

    if rank == target_rank:
        gather_data = []
        for src_rank in range(world_size):
            if src_rank == target_rank:
                gather_data.append(data)
                continue
            recv_data = pipes[(rank, src_rank)].recv()
            gather_data.append(recv_data)
        data = op_func(gather_data)
        return data
    else:
        pipes[(rank, target_rank)].send(data)
        return None


def broadcast_p2p_version(data, source_rank: int = 0):
    # A simplified version of broadcast using point-to-point communication
    # The source rank sends data to all other ranks, O(n) time complexity
    rank = get_rank()
    world_size = get_world_size()
    queue = get_queue()

    if rank == source_rank:
        for _ in range(world_size - 1):
            queue.put(data)
        return data
    else:
        data = queue.get()
        return data


def broadcast(data, source_rank: int = 0):
    # Iterative halving/doubling algorithm for broadcast, O(log(n)) time complexity
    # reference in gloo:
    # https://github.com/facebookincubator/gloo/blob/81925d1c674c34f0dc34dd9a0f2151c1b6f701eb/gloo/broadcast.cc#L20
    rank = get_rank()
    world_size = get_world_size()
    v_rank = (rank - source_rank) % world_size
    pipes = get_pipes()

    iterations = math.ceil(math.log2(world_size))

    involved_rank = 1
    for turn in range(iterations):
        # 1st turn: rank0 -> rank1
        # 2nd turn: rank0 -> rank2, rank1 -> rank3, and so on.
        min_recv_rank = involved_rank  # 2 ** turn
        involved_rank <<= 1
        if v_rank < involved_rank:
            is_send_rank = v_rank < min_recv_rank
            v_peer_rank = (v_rank + min_recv_rank) % involved_rank  # (A, B) to comm, A's peer is B, B's peer is A
            peer_rank = (v_peer_rank + source_rank) % world_size
            if is_send_rank:
                pipes[(rank, peer_rank)].send(data)
            else:
                data = pipes[(rank, peer_rank)].recv()
        else:  # not involved in this turn
            continue

    return data


def all_gather(data):
    """All gather = Gather + Broadcast"""
    data = gather(data)
    # barrier()
    data = broadcast(data)
    return data


def all_reduce(data, op: str = "sum"):
    """All reduce = Reduce + Broadcast"""
    data = reduce(data, op=op)
    # barrier()
    data = broadcast(data)
    return data


def ring_all_reduce(data):
    pass


def mpi_frame(f, world_size: int = 4):
    # import platform
    # assert platform.system() == "Linux", "The script only works on Linux."

    queue = multiprocessing.Queue()
    shared_mem = multiprocessing.Value("i", 0)
    pipe_pairs = make_p2p_pipe(world_size)

    processes = []
    for rank in range(world_size):
        p = multiprocessing.Process(
            target=f,
            args=(rank, world_size, queue, shared_mem, pipe_pairs),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


def test_barrier(rank, world_size, queue, signal_queue, pipe_pairs):
    init_env(rank, world_size, queue, signal_queue, pipe_pairs)

    time.sleep(0.01 * rank)
    print(f"rank{rank} enter func")

    time.sleep(2 * rank)
    print(f"rank{rank} enter barrier")
    barrier()
    # barrier_brooks_algorithm()

    time.sleep(0.01 * rank)
    print(f"Rank{rank} leave.")


def test_broadcast(rank, world_size, queue, signal_queue, pipe_pairs):
    init_env(rank, world_size, queue, signal_queue, pipe_pairs)
    source_rank = 1

    data = [10 * x for x in range(world_size)] if rank == source_rank else None

    time.sleep(0.0001 * rank)
    print(f"Previous data: {data}")
    time.sleep(0.01)

    data = broadcast(data, source_rank=source_rank)

    time.sleep(0.01 * rank)
    print(f"Rank {rank} data: {data}")


def test_reduce(rank, world_size, queue, signal_queue, pipe_pairs):
    init_env(rank, world_size, queue, signal_queue, pipe_pairs)
    source_rank = 0

    data = rank * 10

    time.sleep(0.0001 * rank)
    print(f"Previous data: {data}")
    time.sleep(0.01)

    data = reduce(data, target_rank=source_rank, op="mean")

    time.sleep(0.01 * rank)
    print(f"Rank {rank} data: {data}")


def test_scatter(rank, world_size, queue, signal_queue, pipe_pairs):
    init_env(rank, world_size, queue, signal_queue, pipe_pairs)
    source_rank = 0
    data = [10 * x for x in range(world_size)] if rank == source_rank else None

    time.sleep(0.0001 * rank)
    print(f"Previous data: {data}")
    time.sleep(0.01)

    data = scatter(data, source_rank=source_rank)

    time.sleep(0.01 * rank)
    print(f"Rank {rank} data: {data}")


def test_all_gather(rank, world_size, queue, signal_queue, pipe_pairs):
    init_env(rank, world_size, queue, signal_queue, pipe_pairs)
    data = rank * 10

    time.sleep(0.0001 * rank)
    print(f"Previous data: {data}")
    time.sleep(0.01)

    data = all_gather(data)

    time.sleep(0.01 * rank)
    print("Rank", rank, "data:", data)


def test_all_reduce(rank, world_size, queue, signal_queue, pipe_pairs):
    init_env(rank, world_size, queue, signal_queue, pipe_pairs)
    data = rank * 10

    time.sleep(0.0001 * rank)
    print(f"Previous data: {data}")
    time.sleep(0.01)

    data = all_reduce(data, op="mean")

    time.sleep(0.01 * rank)
    print(f"Rank {rank} data: {data}")


if __name__ == "__main__":
    mpi_frame(test_barrier)
    mpi_frame(test_broadcast)
    mpi_frame(test_reduce)
    mpi_frame(test_scatter)
    mpi_frame(test_all_gather)
    mpi_frame(test_all_reduce)
