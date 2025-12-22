import modal
import torch
import torch.distributed as dist
import os

app = modal.App("pytorch-distributed")

image = modal.Image.debian_slim().pip_install("torch", "numpy")


MATRIX_NUM_ROWS = 1024
MATRIX_NUM_COLS = 1024




def train(rank, world_size):
    # Set up environment for distributed training
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    
    # Initialize process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    ############# INPUT TENSORS SETUP #############
    # Each GPU has:
    #   A_shard: [M, K/world_size] - sharded along K dimension (columns)
    #   B_shard: [K/world_size, N] - sharded along K dimension (rows)
    device = torch.device(f"cuda:{rank}")

    chunk_size = MATRIX_NUM_COLS // world_size
    remainder = MATRIX_NUM_COLS % world_size

    # Last rank gets extra elements if there's a remainder
    if rank < world_size - 1:
        k_start = rank * chunk_size
        k_end = (rank + 1) * chunk_size
    else:
        k_start = rank * chunk_size
        k_end = MATRIX_NUM_COLS  # Last rank gets remainder

    A_shard = torch.ones([MATRIX_NUM_ROWS, k_end - k_start], device=device) * (rank + 1)
    B_shard = torch.ones([k_end - k_start, MATRIX_NUM_ROWS], device=device) * (rank + 1)

    # This simple division only works if you know the MATRIX_NUM_COLS are divisible by world_size!
    # A_shard = torch.ones([MATRIX_NUM_ROWS, MATRIX_NUM_COLS // world_size], device=device) * (rank + 1)
    # B_shard = torch.ones([MATRIX_NUM_COLS // world_size, MATRIX_NUM_ROWS], device=device) * (rank + 1)





    ############# CORE OPERATION #############

    # This is a naive implementation of a sharded GEMM:
    # 1) Each rank computes a *partial* C_local = A_shard @ B_shard
    # 2) reduce-scatter SUM combines partial C_local across ranks and gives each rank *one shard*

    # NOTE: `reduce_scatter_tensor` splits the *input tensor along dim=0* into `world_size` equal chunks.
    #        Will need to transpose this if you want to shard along the columns instead
    C_local = A_shard @ B_shard  # [M, N] partial result on each rank

    N = C_local.shape[1]
    if N % world_size != 0:
        raise ValueError(f"N={N} must be divisible by world_size={world_size} for naive reduce_scatter_tensor")

    # Reduce-scatter across the N dimension (columns) by transposing to [N, M]
    C_local_t = C_local.t().contiguous()  # [N, M]
    out_t = torch.empty((N // world_size, C_local.shape[0]), device=device, dtype=C_local.dtype)  # [N/world, M]
    dist.reduce_scatter_tensor(out_t, C_local_t, op=dist.ReduceOp.SUM)          # BLOCKING: each rank enters the collective, so it can't make progress until all ranks have entered it.
    C_shard = out_t.t().contiguous()  # [M, N/world]

    # REDUCE SCATTER EXPLAINED
    # Before:
        # GPU 0: [C0_chunk, C1_chunk, C2_chunk, C3_chunk]  (partial results)
        # GPU 1: [C0_chunk, C1_chunk, C2_chunk, C3_chunk]
        # GPU 2: [C0_chunk, C1_chunk, C2_chunk, C3_chunk]
        # GPU 3: [C0_chunk, C1_chunk, C2_chunk, C3_chunk]
    # After:
        # GPU 0: [sum(C0_chunk)]  ← Gets chunk 0
        # GPU 1: [sum(C1_chunk)]  ← Gets chunk 1
        # GPU 2: [sum(C2_chunk)]  ← Gets chunk 2
        # GPU 3: [sum(C3_chunk)]  ← Gets chunk 3



    ############# FASTER SOLUTION IDEAS #############
    # reduce-scatter in this case is already highly optimized
    # Higher-perf version is: compute C_local in chunks, then launch reduce_scatter(..., async_op=True) per chunk while computing the next chunk
    #       NVSHMEM isn't the fastest thing here: a pipelined chunked GEMM + async reduce-scatter would be better
    # UVA lets us do remote memory access (higher latency). Not useful in this case, using collectives. Would want to compute tiles, then async communicate.

    ############# Correctness Check #############
    # Given our construction:
    #   A_shard[:] = (rank+1), B_shard[:] = (rank+1)
    # Local GEMM produces a constant matrix with value: k_width(rank) * (rank+1)^2
    # Reduce-scatter SUM should therefore produce a constant C_shard with value:
    #   sum_r k_width(r) * (r+1)^2
    dist.barrier()
    base = MATRIX_NUM_COLS // world_size
    rem = MATRIX_NUM_COLS % world_size
    expected_scalar = base * sum((r + 1) ** 2 for r in range(world_size))
    if rem:
        # Last rank (r = world_size-1) has k_width = base + rem instead of base
        expected_scalar += rem * (world_size ** 2)

    max_err = (C_shard - expected_scalar).abs().max().item()
    if rank == 0:
        print(
            f"C_local shape: {tuple(C_local.shape)} -> C_shard shape per rank: {tuple(C_shard.shape)} | "
            f"expected={expected_scalar} max_err={max_err}"
        )
    assert max_err < 1e-3, f"Reduce-scatter GEMM check failed: max_err={max_err}"



    # Save tensor
    # os.makedirs("/tmp/results", exist_ok=True)
    # torch.save(C_shard, f"/tmp/results/C_shard_rank_{rank}.pt")
    
    dist.destroy_process_group()




@app.function(
    image=image,
    gpu="A100-40GB:8",
    timeout=300,
)
def run_distributed():
    world_size = 8
    torch.multiprocessing.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

@app.local_entrypoint()
def main():
    run_distributed.remote()
