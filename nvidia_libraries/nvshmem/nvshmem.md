# NVSHMEM

Partitioned into a PGAS. One sided communication: just do a put and get.

Interoperate with MPI, can interleave communication and computation at a very fine-grained level.

Whole point is to do in-kernel communication.

- Again you can initialize using MPI:
```c
MPI_Init(&argc, &argv);
MPI_Comm mpi_comm = MPI_COMM_WORLD;

nvshmemx_init_attr_t attr;
nvshmemx_init_attr(...);
...
```
- Can also do MPI broadcast of UID
- Can also interoperate with SHMEM standard's shmem_init()

REVIEW JACOBI SOLVER: using a sync variable and atomic increments n