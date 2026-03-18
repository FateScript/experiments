## Algorithm

algorithm is a collection of algorithm implementations.

### ✨ Intro

**Sorting Algorithms:**

- `bitonic.py` - Bitonic sort algorithm, suitable for parallel-friendly devices such as GPU.
- `sleep_sort.py` - Sleep sort algorithm, a humorous sorting algorithm using multithreading.

**Data Structures:**

- `fenwick_tree.cpp` - Fenwick Tree (Binary Indexed Tree) implementation in C++, supporting prefix sum and range sum queries.

**Sampling:**

- `a_res.py` - A-Res algorithm for weighted random sampling.

**Recursion & Mathematics:**

- `ackermann.py` - Ackermann function implementation, demonstrating the difference between computable and primitive recursive functions, with caching support.
- `recursive.py` - Demonstrates converting mutual recursion (functions `f` and `g`) to non-recursive form using an explicit stack.

**Simulations:**

- `elo_rating.py` - Elo rating system simulation, including player rating convergence, K-value effects, and history-based bonus adjustments.
- `load_balance.py` - Visualization of the "power of two random choices" load balancing algorithm.

### 🛠️ Execute

Executing is simple: using `python3`/`python` command.

For example, to run `bitonic.py`, you could run the following command:
```shell
python3 bitonic.py
```

For C++ files, compile and run:
```shell
g++ -o fenwick_tree fenwick_tree.cpp && ./fenwick_tree
```

Some scripts use `fire` for CLI interfaces. For example:
```shell
python3 ackermann.py --m=3 --n=4
python3 elo_rating.py system
python3 load_balance.py --row=4 --column=5
```
