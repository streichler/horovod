#if HAVE_NCCL
#include "cuda_runtime.h"
#include "nccl.h"
#endif

#include "mpi.h"

#if HAVE_NCCL
void hybridAllReduce(const void* sbuf, void* rbuf, size_t count, ncclDataType_t nccl_type, ncclRedOp_t nccl_op, ncclComm_t nccl_local_comm,
  cudaStream_t stream, void* rbuf_h, MPI_Datatype mpi_type, MPI_Op mpi_op, MPI_Comm local_comm, MPI_Comm node_comm, int lrank, int nsize);

void hybridAllReduce_nosplit(const void* sbuf, void* rbuf, size_t count, ncclDataType_t nccl_type, ncclRedOp_t nccl_op, ncclComm_t nccl_local_comm,
  cudaStream_t stream, void* rbuf_h, MPI_Datatype mpi_type, MPI_Op mpi_op, MPI_Comm local_comm, MPI_Comm node_comm, int lrank, int nsize);
#endif

void hybridAllReduce_nosplit_cpu(const void* sbuf, void* rbuf, size_t count,
    MPI_Datatype mpi_type, MPI_Op mpi_op, MPI_Comm local_comm, MPI_Comm node_comm, int lrank, int nsize);
