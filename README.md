# Guide to Multi-GPU Programming

Unfortunately, there is no well-established single resource/textbook as of writing that I could find that really goes through how to do multi-GPU programming well: it's spread across so many different sources. This is a repo that just compiles a bunch of examples and resources to help teach people how to program multiple GPUs.

Right now, there are so many libraries, frameworks, and concepts that it's hard to keep track of. But this is my opinionated take on the best order in which to learn things.


## Torch Distributed + NCCL
If you're just running experiments and don't care as much about high performance, just use torch distributed with nccl as a backend (examples in the torch_nccl folder).

Minimum Example:
```c
//code here
```

Running:
```bash
./run_my_program
```



## Official NVIDIA Libraries
These have the most documentation and tutorials online, making them the *"easiest" to learn and use*. The tradeoff is they prioritize developer simplicity and *don't maximize performance*. For most non-experts though, they are good enough and have direct NVIDIA support and updates.

The taxonomy is based on GPU Mode's [Intro to MultiGPU Programming](https://www.youtube.com/watch?v=bqL1WC3AKNA):
<details>
  <summary>CUDA-Aware MPI</summary>
  Good if you have existing MPI, with GPU buffers. 
</details>
<details>
  <summary>NCCL</summary>
  If you need stream awareness for high performance collectives
</details>
<details>
  <summary>NVSHMEM</summary>
  One-sided in-kernel communication model. Also supports stream.
</details>




## Triton / ByteDance Libraries
Pythonic device-side DSLs, but might not be super optimized for H100 (only H800).
TileLink: https://arxiv.org/pdf/2503.20313
Triton Distributed: -----put-here-----



## ParallelKittens
LCSC model over the intranode UVA methods
Blog Post: https://hazyresearch.stanford.edu/blog/2025-11-17-pk



## Intranode UVA methods
1. Copy Engine
2. TMA peer access
3. Register instructions for peer access