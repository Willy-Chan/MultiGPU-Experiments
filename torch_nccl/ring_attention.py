import modal
import torch
import torch.distributed as dist
import os

app = modal.App("pytorch-distributed")

image = modal.Image.debian_slim().pip_install("torch", "numpy")


# Ring Attention workload sizes
SEQ_LEN_PER_GPU = 4096      # sequence length (tokens) per GPU chunk
HIDDEN_SIZE = 1024          # token embedding / hidden dimension size
NUM_HEADS = 16              # number of attention heads
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # dimension per attention head (64)
BATCH_SIZE = 1              # batch size (can be increased later)



# FORWARD DISPATCH
#   Can overlap: start processing tokens for expert 0 while still receiving tokens for expert 1 instead of waiting for all2all to finish
#   Avoid having to pack into the all2all and doing the unpacking
#   Better if one expert has a bigger load: don't want to wait for the slowest expert toc complete


# Expert Parallel: inout tokens are the same across all GPUs.
# Current code does EP + DP: input tokens are different across all GPUs.


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

    if HIDDEN_SIZE % NUM_HEADS != 0:
        raise ValueError(f"HIDDEN_SIZE={HIDDEN_SIZE} must be divisible by NUM_HEADS={NUM_HEADS}")

    ############# INPUT TENSORS SETUP (RING ATTENTION) #############
    
    # GPU 0 gets tokens [0:SEQ_LEN_PER_GPU],  GPU 1 gets tokens [SEQ_LEN_PER_GPU:2*SEQ_LEN_PER_GPU], etc.
    
    # Input embeddings (i.e. tokens) for each GPU
    # Shape: [batch_size, seq_len_per_gpu, hidden_size]   --->   [ [FIRST_SET_OF_TOKEN_VECTORS_IN_THE_BATCH], [SECOND_SET_OF_TOKEN_VECTORS_IN_THE_BATCH], etc. ]
    input_embeddings = torch.randn(
        BATCH_SIZE, 
        SEQ_LEN_PER_GPU, 
        HIDDEN_SIZE,
        device=device,
        generator=g
    )
    # **add positional embeddings to the input embeddings here: skipping for now**
    print(f"[Rank {rank}] Input embeddings shape: {input_embeddings.shape}")



    # Q/K/V weights are 2D matrices. THESE WEIGHTS ARE SHARED FOR EACH RANK IN OUR SPECIFIC SETUP:
    #   rank 0 broadcasts the same weights to all other ranks.
    if rank == 0:
        Wq = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device=device) / (HIDDEN_SIZE ** 0.5)
        Wk = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device=device) / (HIDDEN_SIZE ** 0.5)
        Wv = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device=device) / (HIDDEN_SIZE ** 0.5)
    else:
        Wq = torch.empty(HIDDEN_SIZE, HIDDEN_SIZE, device=device)
        Wk = torch.empty(HIDDEN_SIZE, HIDDEN_SIZE, device=device)
        Wv = torch.empty(HIDDEN_SIZE, HIDDEN_SIZE, device=device)
    dist.broadcast(Wq, src=0)
    dist.broadcast(Wk, src=0)
    dist.broadcast(Wv, src=0)



    # q/k/v have the SAME SHAPE as input_embeddings, but different VALUES:
    # they're the result of projecting each token vector through Wq/Wk/Wv.
    q = input_embeddings @ Wq  # [B, seq_len_per_gpu, H]
    k = input_embeddings @ Wk  # [B, seq_len_per_gpu, H]
    v = input_embeddings @ Wv  # [B, seq_len_per_gpu, H]


    
    # Split H into [num_heads, head_dim] and move heads up front for attention math.
    # 
    # Structure for one batch element [b]:
    #   rank_query_matrix[b] has shape [NUM_HEADS, SEQ_LEN_PER_GPU, HEAD_DIM]
    #   - Each head h: rank_query_matrix[b, h] has shape [SEQ_LEN_PER_GPU, HEAD_DIM]
    #   - Each token s in head h: rank_query_matrix[b, h, s] has shape [HEAD_DIM] (64 elements)
    # 
    # Note: The original token had HIDDEN_SIZE=1024 elements, but after splitting into 16 heads,
    # each head only sees HEAD_DIM=64 elements per token.
    # Final shape: [B, num_heads, S_local, head_dim]
    rank_query_matrix = q.view(BATCH_SIZE, SEQ_LEN_PER_GPU, NUM_HEADS, HEAD_DIM).transpose(1, 2).contiguous()
    rank_key_matrix = k.view(BATCH_SIZE, SEQ_LEN_PER_GPU, NUM_HEADS, HEAD_DIM).transpose(1, 2).contiguous()
    rank_value_matrix = v.view(BATCH_SIZE, SEQ_LEN_PER_GPU, NUM_HEADS, HEAD_DIM).transpose(1, 2).contiguous()
    
    print(f"[Rank {rank}] Q shape: {rank_query_matrix.shape}, K shape: {rank_key_matrix.shape}, V shape: {rank_value_matrix.shape}")








    # ############# CORE OPERATION: RING ATTENTION #############
    
    # GOAL: Each GPU needs to compute attention between its local queries (Q) 
    # and ALL keys/values (K/V) from the entire sequence (distributed across all GPUs).
    #
    # RING TOPOLOGY: GPUs are arranged in a ring (0 -> 1 -> 2 -> ... -> 7 -> 0)
    # Each GPU passes its K/V to the next GPU and receives K/V from the previous GPU.
    #
    # ALGORITHM:
    # 1. Each GPU starts with its local Q, K, V
    # 2. For world_size steps:
    #    a. Compute attention: Q_local @ K_current^T (attention scores)
    #    b. Apply softmax to get attention weights
    #    c. Multiply by V_current to get weighted values
    #    d. Accumulate the output
    #    e. Send K, V to next GPU (rank+1)
    #    f. Receive K, V from previous GPU (rank-1)
    # 3. After world_size steps, each GPU has computed attention with all K/V blocks
    
    # Initialize output accumulator for each head
    # Shape: [B, num_heads, seq_len_per_gpu, head_dim]
    output_accumulator = torch.zeros_like(rank_query_matrix)
    
    # Initialize running max and sum for numerical stability in softmax
    # We'll use the online softmax algorithm (more stable for long sequences)
    # These track: max attention score and sum of exp(score - max) per query token
    running_max = torch.full(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN_PER_GPU), 
        float('-inf'), 
        device=device
    )  # Shape: [B, num_heads, seq_len_per_gpu]
    running_sum = torch.zeros(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN_PER_GPU), 
        device=device
    )  # Shape: [B, num_heads, seq_len_per_gpu]
    
    # Current K and V that this GPU is working with
    # Start with this GPU's own K and V
    current_k = rank_key_matrix.clone()  # [B, num_heads, seq_len_per_gpu, head_dim]
    current_v = rank_value_matrix.clone()  # [B, num_heads, seq_len_per_gpu, head_dim]
    
    # Ring topology: determine next and previous ranks
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size
    
    print(f"[Rank {rank}] Starting ring attention: next_rank={next_rank}, prev_rank={prev_rank}")
    
    # Main ring attention loop: iterate world_size times
    for step in range(world_size):
        print(f"[Rank {rank}] Step {step}/{world_size-1}: Computing attention with current K/V block")
        
        # STEP 1: Compute attention scores Q @ K^T
        # rank_query_matrix: [B, num_heads, seq_len_per_gpu, head_dim]
        # current_k:         [B, num_heads, seq_len_per_gpu, head_dim]
        # We want: Q[b, h, i, :] @ K[b, h, j, :]^T for all i, j
        # Result: [B, num_heads, seq_len_per_gpu, seq_len_per_gpu]
        attention_scores = torch.matmul(
            rank_query_matrix,  # [B, num_heads, seq_len_per_gpu, head_dim]
            current_k.transpose(-2, -1)  # [B, num_heads, head_dim, seq_len_per_gpu]
        )  # Result: [B, num_heads, seq_len_per_gpu, seq_len_per_gpu]
        
        # Scale by sqrt(head_dim) for stable gradients
        attention_scores = attention_scores / (HEAD_DIM ** 0.5)
        
        # STEP 2: Apply softmax (using online algorithm for numerical stability)
        # For each query token, find the max score across all key tokens
        # Shape: [B, num_heads, seq_len_per_gpu]
        step_max = attention_scores.max(dim=-1, keepdim=True)[0]  # Max over key dimension
        
        # Update running max: take the maximum of current max and step max
        # This tracks the global max we've seen so far across all K/V blocks
        new_max = torch.maximum(running_max, step_max.squeeze(-1))
        
        # Compute exp(score - new_max) for this step
        # This is the unnormalized attention weights for this K/V block
        exp_scores = torch.exp(attention_scores - new_max.unsqueeze(-1))
        
        # Update running sum using the online softmax formula
        # old_sum * exp(old_max - new_max) + new_sum
        old_max_diff = torch.exp(running_max - new_max)
        running_sum = running_sum * old_max_diff.unsqueeze(-1) + exp_scores.sum(dim=-1, keepdim=True)
        running_max = new_max
        
        # STEP 3: Compute weighted values: attention_weights @ V
        # exp_scores: [B, num_heads, seq_len_per_gpu, seq_len_per_gpu] (unnormalized)
        # current_v:  [B, num_heads, seq_len_per_gpu, head_dim]
        # Result: [B, num_heads, seq_len_per_gpu, head_dim]
        weighted_values = torch.matmul(exp_scores, current_v)
        
        # STEP 4: Accumulate the weighted values into output
        # We'll normalize at the end, so accumulate the unnormalized weighted values
        output_accumulator = output_accumulator * old_max_diff.unsqueeze(-1).unsqueeze(-1) + weighted_values
        
        # STEP 5: Send current K and V to next GPU in ring
        # (except on last step, no need to send)
        if step < world_size - 1:
            # Create send/receive buffers
            send_k = current_k.clone()
            send_v = current_v.clone()
            recv_k = torch.empty_like(current_k)
            recv_v = torch.empty_like(current_v)
            
            # Non-blocking send to next rank
            send_req_k = dist.isend(send_k, dst=next_rank)
            send_req_v = dist.isend(send_v, dst=next_rank)
            
            # Blocking receive from previous rank
            dist.recv(recv_k, src=prev_rank)
            dist.recv(recv_v, src=prev_rank)
            
            # Wait for sends to complete
            send_req_k.wait()
            send_req_v.wait()
            
            # Update current K/V for next iteration
            current_k = recv_k
            current_v = recv_v
            
            print(f"[Rank {rank}] Step {step}: Sent K/V to rank {next_rank}, received from rank {prev_rank}")
    
    # STEP 6: Final normalization
    # Normalize the accumulated output by the running sum
    # output_accumulator: [B, num_heads, seq_len_per_gpu, head_dim]
    # running_sum: [B, num_heads, seq_len_per_gpu, 1] (after unsqueeze)
    final_output = output_accumulator / running_sum.unsqueeze(-1)
    
    # Final output shape: [B, num_heads, seq_len_per_gpu, head_dim]
    # This is the attention output for this GPU's local queries, having attended to ALL keys/values
    print(f"[Rank {rank}] Ring attention complete! Final output shape: {final_output.shape}")
    
    # Synchronize all GPUs before proceeding
    dist.barrier()
    
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
