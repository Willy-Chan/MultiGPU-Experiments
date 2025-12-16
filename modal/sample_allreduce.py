import modal
import torch
import torch.distributed as dist
import os

app = modal.App("pytorch-distributed")

image = modal.Image.debian_slim().pip_install("torch", "numpy")

def train(rank, world_size):
    # Set up environment for distributed training
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    
    # Initialize process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    # INPUT TENSORS ON EACH GPU    
    device = torch.device(f"cuda:{rank}")
    tensor = torch.ones(10, device=device) * (rank + 1)
    




    print(f"Rank {rank}: Before allreduce: {tensor[:5]}")
    
    # Perform allreduce
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    print(f"Rank {rank}: After allreduce: {tensor[:5]}")
    


    # # Save tensor
    # os.makedirs("/tmp/results", exist_ok=True)
    # torch.save(tensor, f"/tmp/results/tensor_rank_{rank}.pt")
    
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
