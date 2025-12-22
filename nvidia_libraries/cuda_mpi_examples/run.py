# 1. THE HPC IMAGE: Building CUDA-aware OpenMPI + UCX from source
# # This mimics a cluster where MPI is tuned for the specific hardware.
# hpc_image = (
#     modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
#     .apt_install("wget", "build-essential", "libfabric-dev", "libnuma-dev")
#     .run_commands(
#         # Install UCX first (The communication engine for CUDA-aware MPI)
#         "wget https://github.com/openucx/ucx/releases/download/v1.15.0/ucx-1.15.0.tar.gz",
#         "tar xf ucx-1.15.0.tar.gz && cd ucx-1.15.0 && "
#         "./configure --prefix=/opt/ucx --with-cuda=/usr/local/cuda && "
#         "make -j$(nproc) install",
        
#         # Install OpenMPI linked to UCX and CUDA
#         "wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz",
#         "tar xf openmpi-4.1.6.tar.gz && cd openmpi-4.1.6 && "
#         "./configure --prefix=/opt/openmpi --with-cuda=/usr/local/cuda --with-ucx=/opt/ucx && "
#         "make -j$(nproc) install"
#     )
#     .env({
#         "PATH": "/opt/openmpi/bin:/usr/local/cuda/bin:$PATH",
#         "LD_LIBRARY_PATH": "/opt/openmpi/lib:/opt/ucx/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
#         # HPC Best Practice: Disable memory type cache to avoid issues with GPU memory recycling
#         "UCX_MEMTYPE_CACHE": "n", 
#         "OMPI_ALLOW_RUN_AS_ROOT": "1",
#         "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
#     })
#     .add_local_dir(".", remote_path="/root/workspace")
# )


# Results in a lot of UCX errors and warnings, but seems to at least run in my experience.

import modal

hpc_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/nvhpc:24.3-devel-cuda12.3-ubuntu22.04", # This image is the industry standard for HPC. 
        add_python="3.11" # This tells Modal to inject Python into the NVHPC base
    )
    .apt_install("python-is-python3") # Ensures 'python' command works
    .env({
        # Point to the pre-installed OpenMPI within the HPC SDK
        "PATH": "/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/openmpi/openmpi-4.1.6/bin:/usr/local/cuda/bin:$PATH",
        "LD_LIBRARY_PATH": "/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/openmpi/openmpi-4.1.6/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        
        # Standard HPC flags for root execution & CUDA support
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
        "UCX_MEMTYPE_CACHE": "n", 
    })
    .add_local_dir(".", remote_path="/root/workspace")
)

app = modal.App("hpc-cluster-simulation")

@app.function(
    image=hpc_image, 
    gpu="A100-40GB:2", 
    timeout=600
)
def run_hpc_job(cuda_file: str):
    import subprocess
    import os
    
    workspace = "/root/workspace"
    os.chdir(workspace)
    
    if not os.path.exists(cuda_file):
        return f"Error: {cuda_file} not found in {workspace}. Check your local directory."

    # 1. DYNAMICALLY DISCOVER MPI FLAGS (The robust way)
    # We split the output by spaces to handle multiple include/lib directories
    try:
        inc_dirs = subprocess.check_output(["mpicc", "--showme:incdirs"], text=True).strip().split()
        lib_dirs = subprocess.check_output(["mpicc", "--showme:libdirs"], text=True).strip().split()
    except subprocess.CalledProcessError:
        return "Error: MPI wrappers not found. Check PATH."

    # Construct individual -I and -L flags
    inc_flags = [f"-I{d}" for d in inc_dirs]
    lib_flags = [f"-L{d}" for d in lib_dirs]

    # 2. COMPILE WITH NVCC
    compile_cmd = ["nvcc", "-O3", cuda_file, "-o", "mpi_app"] + inc_flags + lib_flags + ["-lmpi", "-lcudart", "-lstdc++"]
    
    print(f"--- Compiling: {' '.join(compile_cmd)} ---")
    comp_result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if comp_result.returncode != 0:
        return f"Compilation Failed:\n{comp_result.stderr}"

    # 3. EXECUTE WITH UCX (CUDA-Aware engine)
    run_cmd = [
        "mpirun", "-np", "2",
        "--mca", "pml", "ucx",
        "--mca", "opal_cuda_support", "1",
        "./mpi_app"
    ]
    
    print(f"--- Executing HPC Job ---")
    result = subprocess.run(run_cmd, capture_output=True, text=True)
    return result.stdout + result.stderr

@app.local_entrypoint()
def main(cuda_file: str = "example.cu"):
    print(run_hpc_job.remote(cuda_file))