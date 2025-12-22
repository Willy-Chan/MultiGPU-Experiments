# NCCL

## Summary
In the Jacobi kernel example, MPI_sendrecv doesn't have a stream argument.

So we have to synchronize streams for the top and bottom parts: can't automatically overlap them/have them be at the same time here:

![Can't remove the stream sync](cantremovesync.png)

Ideally these two sendrecvs happen at the same time (different streams), but MPI cant accept this argument!


NCCL basically lets you add this stream support. Think CUDA-aware MPI, but instead of calling MPI we're calling dedicated communication kernels, *with stream support*.

You initialize NCCL by just piggybacking off of MPI: you broadcast the NCCLUid and that's what you need to set everything up. (can use both NCCL and MPI)
```c
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

ncclUniqueId nccl_uid;
if (rank == 0) ncclGetUniqueId(&nccl_uid);
MPI_Bcast(&nccl_uid, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
MPI_Barrier(MPI_COMM_WORLD);

ncclComm_t nccl_comm;
ncclCommInitRank(&nccl_comm, size, nccl_uid, rank);
// DO YOUR STUFF
ncclCommDestroy(nccl_comm);
MPI_Finalize();
```

- Use `ncclGroupStart()` and `ncclGroupEnd()` around nccl operations you want to aggregate: that way you pay the launch overhead only once. (put around multiple ncclAllReduce calls)

```
ncclGroupStart()
ncclSend and ncclRecvs
ncclGroupEnd()
```

- Streams can have different "priorities" 
- NCCL overlap just gives you increased efficiency

- Literally just MPI, but only on device buffers, and it's stream-aware.