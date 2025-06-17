#!/usr/bin/env python3

import sys
import time
import threading


def cpu_task(range_limit: int = 10**7):
    return sum(i * i for i in range(range_limit))


def check_time(run_loops: int = 100):
    start_time = time.time()
    for _ in range(run_loops):
        cpu_task()
    end_time = time.time()
    print(f"Single-threaded execution time: {end_time - start_time} seconds")

    threads = [threading.Thread(target=cpu_task) for _ in range(run_loops)]
    start_time = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    end_time = time.time()
    print(f"Multi-threaded execution time: {end_time - start_time} seconds")


switch_count = 0
last_thread = None


def tracefunc(frame, event, arg):
    global switch_count, last_thread
    current_thread = threading.current_thread().name
    if current_thread != last_thread:
        print(f"Thread switch detected: {last_thread} -> {current_thread}")
        if last_thread is not None and last_thread.startswith("Thread") and current_thread.startswith("Thread"):
            switch_count += 1
        last_thread = current_thread
    return tracefunc


def compute_bound_task():
    sys.settrace(tracefunc)
    print("Starting compute-bound task...")
    cpu_task(range_limit=10**8)
    print(f"Thread {threading.current_thread().name} finished.")


def io_bound_task():
    sys.settrace(tracefunc)
    print("Starting I/O-bound task...")
    time.sleep(2)
    print(f"Thread: {threading.current_thread().name} finished.")


def check_context_switch():
    """Check if the GIL is released by running a CPU-bound task in multiple threads."""
    threads = [
        threading.Thread(target=io_bound_task, name=f"Thread-io-{i}") for i in range(1)
    ] + [
        threading.Thread(target=compute_bound_task, name="Thread-compute"),
    ]
    start_time = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")
    print(f"Thread switches count: {switch_count}")


if __name__ == "__main__":
    # check_time()
    check_context_switch()
