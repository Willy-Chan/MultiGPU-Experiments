import modal
import torch
import torch.distributed as dist
import os

app = modal.App("pytorch-distributed")

image = modal.Image.debian_slim().pip_install("torch", "numpy")


# MoE dispatch toy workload sizes
NUM_TOKENS_PER_GPU = 4096   # tokens initially resident on each GPU (before routing)
HIDDEN_SIZE = 1024          # token embedding / MLP input size
NUM_EXPERTS = 32            # total experts globally (must be divisible by world_size for this toy)
TOP_K = 1                   # routing choices per token (this toy implements TOP_K=1)




def train(rank, world_size):
    # Set up environment for distributed training
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    
    # Initialize process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}")

    # Use a deterministic seed per-rank so runs are stable/reproducible.
    g = torch.Generator(device=device)
    g.manual_seed(1234 + rank)

    # NOTE: This example only implements this for TOP_K == 1, and assumes the number of experts is divisible by the world_size
    if TOP_K != 1:
        raise NotImplementedError("This toy dispatch implements TOP_K=1 only (extend by flattening [T,k]).")
    if NUM_EXPERTS % world_size != 0:
        raise ValueError(f"NUM_EXPERTS={NUM_EXPERTS} must be divisible by world_size={world_size} for this toy.")

    
    ############# INPUT TENSORS SETUP (MoE DISPATCH) #############
    # Each GPU starts with:
    #   x: [T_r, H] token activations local to this GPU
    #   expert_id: [T_r] global expert assignment per token (TOP_K=1 in this toy)
    #
    # Dispatch goal:
    #   1) pack tokens by destination GPU (based on expert_id)
    #   2) all-to-all exchange token activations + expert_id
    #   3) locally regroup received tokens by *local* expert so each expert's tokens are contiguous
    


    experts_per_gpu = NUM_EXPERTS // world_size
    expert_start = rank * experts_per_gpu
    expert_end = expert_start + experts_per_gpu

    # Token activations on this GPU
    x = torch.randn((NUM_TOKENS_PER_GPU, HIDDEN_SIZE), device=device, generator=g)

    # Global expert assignment per token (TOP_K=1): values in [0, NUM_EXPERTS)
    expert_id = torch.randint(
        low=0, high=NUM_EXPERTS, size=(NUM_TOKENS_PER_GPU,), device=device, generator=g, dtype=torch.int64
    )

    # GPU 0 owns experts [0 ... experts_per_gpu-1], GPU 1 owns [experts_per_gpu ... 2 * experts_per_gpu - 1], etc.
    # The destination GPU for each token is   expert_id[token] // experts_per_gpu
    #   For NUM_EXPERTS=32, world_size = 8, experts_per_gpu = 4, we use floor-division here so:
    #       expert_id = 0...3  -> dst_gpu = 0
    #       expert_id = 4...7  -> dst_gpu = 1
    #       expert_id = 8...11  -> dst_gpu = 2
    dst_gpu = expert_id // experts_per_gpu  # [T, top_K]

    # pack_perm is [---token_IDs_belonging_to_GPU_0---, ---token_IDs_belonging_to_GPU_1---, ---token_IDs_belonging_to_GPU_2---]
    pack_perm = torch.argsort(dst_gpu)  # [T]



    # Tokens and expert_ids, in the correct packed order!
    send_x = x.index_select(0, pack_perm).contiguous()                 # [T, H]
    send_expert_id = expert_id.index_select(0, pack_perm).contiguous() # [T]



    # Split sizes for all-to-all
    # [6, 2, 5, 9, 4, 6, 7, 8] - send 6 tokens to gpu 0, etc.
    send_counts = torch.bincount(dst_gpu, minlength=world_size).to(torch.int64)  # [G]
    recv_counts = torch.empty((world_size,), device=device, dtype=torch.int64)   # [0, 0, 0, 0, 0, 0, 0, 0] 
    # Exchange counts elementwise (each rank sends 1 int64 to each other rank)
    # Alltoall of just the counts
    dist.all_to_all_single(recv_counts, send_counts)

    # So now every GPU knows how many tokens its going to receive. We need this info (recv_total = int(sum(recv_splits))) so we can allocate
    # the tensor that will ACTUALLY be getting the tokens in question!
    send_splits = send_counts.cpu().tolist()
    recv_splits = recv_counts.cpu().tolist()
    recv_total = int(sum(recv_splits))

    
    recv_x = torch.empty((recv_total, HIDDEN_SIZE), device=device, dtype=send_x.dtype)  # tensor that will get the tokens
    recv_expert_id = torch.empty((recv_total,), device=device, dtype=torch.int64)       # tensor that will say, for each of these tokens, what expert is it


    # ############# CORE OPERATION (DISPATCH OUTPUT: READY-FOR-MLP BUFFERS) #############

    # ALL-TO-ALL ON BOTH THE TOKENS, AND THE EXPERTIDs
    dist.all_to_all_single(recv_x, send_x, output_split_sizes=recv_splits, input_split_sizes=send_splits)
    dist.all_to_all_single(
        recv_expert_id, send_expert_id, output_split_sizes=recv_splits, input_split_sizes=send_splits
    )

    # Print essential tensor shapes after all-to-all
    dist.barrier()  # Ensure all ranks have completed before printing
    print(f"[rank {rank}] After all-to-all:")
    print(f"  send_x shape: {tuple(send_x.shape)}")
    print(f"  recv_x shape: {tuple(recv_x.shape)}")
    print(f"  send_expert_id shape: {tuple(send_expert_id.shape)}")
    print(f"  recv_expert_id shape: {tuple(recv_expert_id.shape)}")
    print(f"  send_counts: {send_counts.cpu().tolist()}")
    print(f"  recv_counts: {recv_counts.cpu().tolist()}")
    print(f"  recv_total: {recv_total}")
    print()


    # At this point, recv_x contains exactly the tokens whose experts live on this GPU.
    # But we still need to regroup the tokens in recv_x by expert! Luckily all our experts are on this GPU.

    # Regroup by local expert so each expert can run an MLP over a contiguous slice:
    if recv_total > 0:

        # vector of each expert assigned to each token
        local_expert = (recv_expert_id - expert_start).to(torch.int64)  # [recv_total]

        if not bool(((0 <= local_expert) & (local_expert < experts_per_gpu)).all().item()):
            bad = recv_expert_id[(local_expert < 0) | (local_expert >= experts_per_gpu)][:8].tolist()
            raise RuntimeError(
                f"Rank {rank} received tokens for non-local experts. Example global expert ids: {bad} "
                f"(local expert range is [{expert_start}, {expert_end}))"
            )

        # sort the local_expert list
        # [---expert_local_1's token indexes---, ---expert_local_2's token indexes---, etc.]
        expert_perm = torch.argsort(local_expert)   
        # [--------expert_local_1--------------, --------expert_local_2--------------, etc.]
        expert_ids_local_sorted = local_expert.index_select(0, expert_perm).contiguous()  # [N_r], 

        # convert expert_perm to giant matrix of tokens
        # [---expert_local_1's tokens---, ---expert_local_2's tokens---, etc.]
        expert_in = recv_x.index_select(0, expert_perm).contiguous()  # [N_r, H], grouped by local expert

        # [num_tokens local_expert_0 has, num_tokens local_expert_1 has, etc. ]
        tokens_per_expert = torch.bincount(expert_ids_local_sorted, minlength=experts_per_gpu).to(torch.int64)

        # [A, B, C] where expert_in[A, B) contains expert 0's tokens, [B, C) contains expert 1's tokens, etc.
        # use tokens_per_expert and cumsum to calculate this:
        #   cumsum:
        #       t = torch.tensor([1, 2, 3, 4, 5])
        #       result = torch.cumsum(t, dim=0)
        #       result is tensor([ 1,  3,  6, 10, 15])
        expert_offsets = torch.zeros((experts_per_gpu + 1,), device=device, dtype=torch.int64)
        expert_offsets[1:] = torch.cumsum(tokens_per_expert, dim=0)
    else:
        expert_in = recv_x  # empty [0, H]
        tokens_per_expert = torch.zeros((experts_per_gpu,), device=device, dtype=torch.int64)
        expert_offsets = torch.zeros((experts_per_gpu + 1,), device=device, dtype=torch.int64)


    dist.barrier()

    if rank == 0:
        print(
            "MoE dispatch complete (TOP_K=1). "
            f"Each GPU started with x=[{NUM_TOKENS_PER_GPU}, {HIDDEN_SIZE}]. "
            f"Experts per GPU={experts_per_gpu}, total experts={NUM_EXPERTS}."
        )
    print(
        f"[rank {rank}] send_counts={send_splits} recv_counts={recv_splits} "
        f"-> expert_in={tuple(expert_in.shape)} tokens_per_expert(sum)={int(tokens_per_expert.sum().item())}"
    )



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
