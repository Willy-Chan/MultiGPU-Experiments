import modal
import os

# 1. Image Definition
# We use CUDA 12.2 to match Modal's Host Driver.
image = (
    modal.Image.from_registry("nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04", add_python="3.11")
    .apt_install("openmpi-bin", "libopenmpi-dev", "wget")
    .pip_install(
        "torch",            # Triton comes bundled with Torch usually, or we install it separately
        "triton",           # Explicitly install triton
        "mpi4py",
        "nvshmem4py-cu12",
        "cuda-python>=12.0",
    )
    .env({
        "NVSHMEM_REMOTE_TRANSPORT": "p2p",
        "NVSHMEM_DISABLE_P2P": "0",
        "NVIDIA_DISABLE_REQUIRE": "1",
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
    })
)

app = modal.App("nvshmem-triton-p2p", image=image)

@app.function(gpu="A100:2", timeout=600)
def run_nvshmem_test():
    import subprocess
    import tempfile
    
    script_content = """
import torch
import triton
import triton.language as tl
import nvshmem.core as nvshmem
from mpi4py import MPI
from cuda.core.experimental import Device, system

# --- 1. THE TRITON KERNEL ---
# Triton kernels accept pointers as arguments.
# When you pass a torch.Tensor, Triton automatically passes its .data_ptr().
@triton.jit
def p2p_store_kernel(
    ptr,  # Pointer to the destination (peer memory)
    val,  # Value to write
    BLOCK_SIZE: tl.constexpr
):
    # Standard Triton boilerplate to get program ID
    pid = tl.program_id(0)
    
    # We only want one thread to do the write for this simple demo
    if pid == 0:
        # Load the pointer.
        # Note: 'ptr' is just a memory address (int64).
        # We add 0 to it to get the first element address.
        output_ptr = ptr + 0
        
        # Store the value directly to the remote address
        tl.store(output_ptr, val)

# --- 2. SETUP ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
local_rank = rank % system.num_devices

# CRITICAL for Triton: Set the device so the compiler knows where to run
torch.cuda.set_device(local_rank)

# NVSHMEM Setup
dev = Device(local_rank)
dev.set_current()
stream = torch.cuda.current_stream() # Use Torch's default stream for convenience

nvshmem.init(device=dev, mpi_comm=comm, initializer_method="mpi")
my_pe = nvshmem.my_pe()
n_pes = nvshmem.n_pes()

# --- 3. SYMMETRIC ALLOCATION ---
# Allocate a standard PyTorch tensor on the Symmetric Heap
tensor = nvshmem.tensor((1,), dtype=torch.int32)
tensor[0] = -1

# --- 4. GET PEER VIEW ---
dst_pe = (my_pe + 1) % n_pes

# This returns a torch.Tensor that points to the neighbor's memory via UVA.
# No casting needed!
dev_dst = nvshmem.get_peer_tensor(tensor, dst_pe)

print(f"[Rank {my_pe}] Pushing rank to PE {dst_pe} (Addr: {hex(dev_dst.data_ptr())})...")

# --- 5. LAUNCH KERNEL ---
# We pass 'dev_dst' directly. Triton handles the pointer extraction.
grid = (1,)
p2p_store_kernel[grid](dev_dst, my_pe, BLOCK_SIZE=32)

# --- 6. BARRIER & CHECK ---
# We use the NVSHMEM barrier to ensure the P2P write has landed
# We need to wrap the torch stream handle for NVSHMEM
# Create a wrapper class that implements __cuda_stream__
# Example of using https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html#cuda-stream-protocol
class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream
        self.handle = pt_stream.cuda_stream

    def __cuda_stream__(self):
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)  # Return format required by CUDA Python

nvshmem.barrier(nvshmem.Teams.TEAM_NODE, PyTorchStreamWrapper(stream))

print(f"PE {my_pe}: Received value {tensor[0].item()} from neighbor")

nvshmem.free_tensor(tensor)
nvshmem.finalize()
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        env = os.environ.copy()
        env["NVSHMEM_SYMMETRIC_SIZE"] = "1G"
        
        cmd = ["mpirun", "-np", "2", "python3", script_path]
        print(f"Executing: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        print(result.stdout)
        print(result.stderr)
        
        if result.returncode != 0:
            raise Exception("MPI Execution Failed")
            
        return "Success"
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)

@app.local_entrypoint()
def main():
    run_nvshmem_test.remote()