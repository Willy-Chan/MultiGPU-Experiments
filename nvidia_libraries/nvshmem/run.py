import modal
import os

# The HPC SDK includes NVSHMEM in the math_libs directory
image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/nvhpc:24.3-devel-cuda12.3-ubuntu22.04", 
        add_python="3.11"
    )
    .apt_install("python-is-python3")
    .env({
        "PATH": "/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/openmpi/openmpi-4.1.6/bin:/usr/local/cuda/bin:$PATH",
        "LD_LIBRARY_PATH": "/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/openmpi/openmpi-4.1.6/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
    })
    .add_local_dir(".", remote_path="/root/workspace")
)

app = modal.App("nvshmem-runner")

@app.function(
    image=image, 
    gpu="A100-40GB:2", 
    timeout=600
)
def run_nvshmem_code(cuda_file: str):
    import subprocess
    os.chdir("/root/workspace")
    
    # 1. DEFINE NVSHMEM PATHS (HPC SDK standard)
    nvsh_root = "/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/nvshmem"
    nvsh_inc = f"{nvsh_root}/include"
    nvsh_lib = f"{nvsh_root}/lib"

    # 2. DISCOVER MPI FLAGS
    mpi_inc = subprocess.check_output(["mpicc", "--showme:incdirs"], text=True).strip().split()[0]
    mpi_lib = subprocess.check_output(["mpicc", "--showme:libdirs"], text=True).strip().split()[0]

    # 3. COMPILE
    # NVSHMEM requires linking the static library, CUDA driver, and NVML (for P2P detection)
    compile_cmd = [
        "nvcc", "-O3", cuda_file, "-o", "nvsh_app",
        f"-I{nvsh_inc}", f"-I{mpi_inc}",
        f"-L{nvsh_lib}", f"-L{mpi_lib}",
        "-lnvshmem", "-lmpi", "-lcuda", "-lcudart", "-lnvidia-ml"
    ]
    
    print(f"--- Compiling NVSHMEM: {' '.join(compile_cmd)} ---")
    comp_res = subprocess.run(compile_cmd, capture_output=True, text=True)
    if comp_res.returncode != 0:
        return f"Compilation Failed:\n{comp_res.stderr}"

    # 4. RUN
    # We use mpirun to launch. NVSHMEM will detect the MPI environment.
    run_cmd = [
        "mpirun", "-np", "2",
        "-x", "NVSHMEM_REMOTE_TRANSPORT=p2p", # Force Peer-to-Peer instead of IB
        "-x", "NVSHMEM_DISABLE_P2P=0",
        "-x", "UCX_LOG_LEVEL=error",           # Silence UCX warnings
        "--mca", "coll_hcoll_enable", "0",      # Silence HCOLL warnings
        "./nvsh_app"
    ]

    print(f"--- Running NVSHMEM App ---")
    # Setting NVSHMEM_SYMMETRIC_SIZE ensures enough heap for the GPUs to share
    env = os.environ.copy()
    env["NVSHMEM_SYMMETRIC_SIZE"] = "1G" 
    
    run_res = subprocess.run(run_cmd, capture_output=True, text=True, env=env)
    return run_res.stdout + run_res.stderr

@app.local_entrypoint()
def main(cuda_file: str):
    print(run_nvshmem_code.remote(cuda_file))