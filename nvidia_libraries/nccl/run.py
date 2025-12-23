import modal

import modal
import os

# Using the HPC SDK which has NCCL and MPI pre-installed
image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/nvhpc:24.3-devel-cuda12.3-ubuntu22.04", 
        add_python="3.11"
    )
    .apt_install("python-is-python3")
    .env({
        "PATH": "/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/openmpi/openmpi-4.1.6/bin:/usr/local/cuda/bin:$PATH",
        "LD_LIBRARY_PATH": "/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/openmpi/openmpi-4.1.6/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        
        # NCCL Specific Tweak: Sometimes containers have small /dev/shm. 
        # This forces NCCL to use P2P/Shared memory correctly.
        "NCCL_P2P_DISABLE": "0", 
        "NCCL_SHM_DISABLE": "0",
        
        # MPI flags needed to bootstrap the NCCL processes
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
    })
    .add_local_dir(".", remote_path="/root/workspace")
)

app = modal.App("nccl-runner")

@app.function(
    image=image, 
    gpu="A100-40GB:2",  # You need at least 2 GPUs to see NCCL work
    timeout=600
)
def run_nccl_code(cuda_file: str):
    import subprocess
    
    os.chdir("/root/workspace")
    
    # 1. DISCOVER NCCL PATHS
    # In the HPC SDK, NCCL is located in the math_libs folder
    nccl_base = "/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/math_libs/12.3/nccl"
    nccl_inc = f"{nccl_base}/include"
    nccl_lib = f"{nccl_base}/lib"

    # 2. DISCOVER MPI PATHS (for bootstrapping)
    mpi_inc = subprocess.check_output(["mpicc", "--showme:incdirs"], text=True).strip().split()[0]
    mpi_lib = subprocess.check_output(["mpicc", "--showme:libdirs"], text=True).strip().split()[0]

    # 3. COMPILE
    # We link both MPI (for the control plane) and NCCL (for the data plane)
    compile_cmd = [
        "nvcc", "-O3", cuda_file, "-o", "nccl_app",
        f"-I{nccl_inc}", f"-I{mpi_inc}",
        f"-L{nccl_lib}", f"-L{mpi_lib}",
        "-lnccl", "-lmpi", "-lcudart"
    ]
    
    print(f"--- Compiling: {' '.join(compile_cmd)} ---")
    comp_res = subprocess.run(compile_cmd, capture_output=True, text=True)
    if comp_res.returncode != 0:
        return f"Compilation Failed:\n{comp_res.stderr}"

    # 4. RUN
    # We use mpirun to start the processes, which then initialize NCCL
    run_cmd = [
        "mpirun", "-np", "2",
        "--mca", "opal_cuda_support", "1",
        "./nccl_app"
    ]
    
    print(f"--- Running NCCL App ---")
    run_res = subprocess.run(run_cmd, capture_output=True, text=True)
    return run_res.stdout + run_res.stderr

@app.local_entrypoint()
def main(cuda_file: str):
    print(run_nccl_code.remote(cuda_file))