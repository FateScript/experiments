#!/usr/bin/env python3

import concurrent.futures
import math
import multiprocessing as mp
import tempfile
import timeit


def cpu_bound_work(n):
    return sum(math.factorial(i) for i in range(n))


def io_bound_work():
    temp_file = tempfile.NamedTemporaryFile(delete=True)
    filename = temp_file.name

    content = "This is a test content." * 1000
    with open(filename, 'w') as f:
        f.write(content)

    with open(filename, 'r') as f:
        return f.read()


def concurrent_thread(f, *args, num_workers: int = 8, **kwargs):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(f, *args, **kwargs) for _ in range(num_workers)]
        return [f.result() for f in futures]


def concurrent_process(f, *args, num_workers: int = 8, **kwargs):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(f, *args, **kwargs) for _ in range(num_workers)]
        return [f.result() for f in futures]


def benchmark_cpu_bound(n: int = 100, number: int = 100):
    local_dict = locals()
    global_dict = globals()
    global_dict.update(local_dict)

    thread_time = timeit.timeit(
        'concurrent_thread(cpu_bound_work, n)',
        globals=global_dict, number=number,
        # setup=f"n = {n}"
    )
    print(f"ThreadPoolExecutor time: {thread_time} seconds")

    process_time = timeit.timeit(
        'concurrent_process(cpu_bound_work, n)',
        globals=global_dict, number=number,
    )
    print(f"ProcessPoolExecutor time: {process_time} seconds")


def benchmark_io_bound(number: int = 10):
    thread_time = timeit.timeit(
        'concurrent_thread(io_bound_work)',
        globals=globals(),
        number=number,
    )
    print(f"ThreadPoolExecutor time: {thread_time} seconds")

    process_time = timeit.timeit(
        'concurrent_process(io_bound_work)',
        globals=globals(),
        number=number,
    )
    print(f"ProcessPoolExecutor time: {process_time} seconds")


if __name__ == "__main__":
    start_method = mp.get_start_method()
    print(f"mp start method: {start_method}")

    # Python's GIL allows only one thread to execute Python bytecode at a time.
    # For I/O-bound tasks, this isn’t a major issue
    # because threads can yield to one another during I/O operations.
    # For CPU-bound tasks, this can limit the effectiveness of threading in Python
    # because the GIL can prevent multiple threads from fully utilizing multiple CPU cores.

    # In summary, for I/O-bound tasks, please use multi-thread.
    # For CPU-bound tasks, please use multi-process.

    print("\nbenchmark fake cpu bound")
    # Creating and managing processes is more expensive than threads.
    # the tasks are short or lightweight, overhead of process creation and IPC can dominate.
    benchmark_cpu_bound(100, 100)

    print("\nbenchmark real cpu bound")
    benchmark_cpu_bound(5000, 3)

    print("\nbenchmark io bound")
    benchmark_io_bound(10)
