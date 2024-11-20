
### MPI

mpi implements algorithms using MPI for parallel computing.

### ✨ Intro

`mpi.py`: basic MPI functions, such as `reduce`, `gather`, `broadcast`, etc.

`linear_mpi.py`: a numpy version torch.nn.Linear class using numpy with Row/Column Parallel.

`mha_mpi.py`: a numpy verision Multi-Head Attention class with Sequence/Tensor Parallel.

`megatron_sp.py`: reimplement Megatron Sequence Parallel proposed in Megatron-LM. [paper](https://arxiv.org/pdf/2205.05198)
