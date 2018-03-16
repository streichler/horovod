#include "cuda_runtime.h"
#include "nccl.h"

#include "mpi.h"

void hybridAllReduce(const float* sbuf, float* rbuf, size_t count, ncclComm_t nccl_local_comm,
  cudaStream_t stream, float* rbuf_h, MPI_Comm local_comm, MPI_Comm node_comm, int lrank, int nsize);

void hybridAllReduce_2split(const float* sbuf, float* rbuf, size_t count, ncclComm_t nccl_local_comm,
  cudaStream_t stream, float* rbuf_h, MPI_Comm local_comm, MPI_Comm node_comm, int lrank, int nsize);

void hybridAllReduce_nosplit(const float* sbuf, float* rbuf, size_t count, ncclComm_t nccl_local_comm,
  cudaStream_t stream, float* rbuf_h, MPI_Comm local_comm, MPI_Comm node_comm, int lrank, int nsize);
