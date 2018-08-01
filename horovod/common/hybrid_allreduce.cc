#if HAVE_NCCL
#include "cuda_runtime.h"
#include "nccl.h"
#endif

#include "mpi.h"

#include "macros.h"
#include "hybrid_allreduce.h"

#define ALIGN_FLOATS 256

#if HAVE_NCCL
void hybridAllReduce(const void* sbuf, void* rbuf, size_t count, ncclDataType_t nccl_type, ncclRedOp_t nccl_op, ncclComm_t nccl_local_comm,
  cudaStream_t stream, void* rbuf_h, MPI_Datatype mpi_type, MPI_Op mpi_op, MPI_Comm local_comm, MPI_Comm node_comm, int lrank, int nsize)
{
  int typesize;
  MPI_Type_size(mpi_type, &typesize);

  /* AllReduce on node */
  NCCLCHECK(ncclAllReduce(sbuf, rbuf, count, nccl_type, nccl_op, nccl_local_comm, stream));

  if (nsize > 1)
  {
    size_t blockcount = count/ALIGN_FLOATS/4 * ALIGN_FLOATS; 

    // If buffer is too small to split, fallback to no split version
    if (blockcount == 0)
    {
       hybridAllReduce_nosplit(sbuf, rbuf, count, nccl_type, nccl_op, nccl_local_comm,
         stream, rbuf_h, mpi_type, mpi_op, local_comm, node_comm, lrank, nsize);
       return;
    }

    /* Intranode AllReduce using local ranks 0,1,3,4*/
    if (lrank == 0 or lrank == 1 or lrank == 3 or lrank == 4)
    {
      size_t shift = 0;
      if (lrank == 1) shift = 1;
      else if (lrank == 3) shift = 2;
      else if (lrank == 4) shift = 3;

      

      CUDACHECK(cudaMemcpyAsync(rbuf_h, rbuf + (shift * blockcount)*typesize, ((lrank == 4) ? blockcount + count%blockcount : blockcount)*typesize,
                  cudaMemcpyDeviceToHost, stream));
      cudaStreamSynchronize(stream);

      PUSH_RANGE("MPI_Allreduce", 0)
      MPI_Allreduce(MPI_IN_PLACE, rbuf_h, ((lrank == 4) ? blockcount + count%blockcount : blockcount), mpi_type, mpi_op, node_comm);
      POP_RANGE

      CUDACHECK(cudaMemcpyAsync(rbuf + (shift * blockcount)*typesize, rbuf_h, ((lrank == 4) ? blockcount + count%blockcount : blockcount)*typesize,
                  cudaMemcpyHostToDevice, stream));
      //cudaStreamSynchronize(stream);
    }

    MPI_Barrier(local_comm);
    //MPI_Barrier(node_comm);

    /* Bcast on node */
    NCCLCHECK(ncclBcast(rbuf, blockcount, nccl_type, 0, nccl_local_comm, stream));
    NCCLCHECK(ncclBcast(rbuf + (1 * blockcount)*typesize, blockcount, nccl_type, 1, nccl_local_comm, stream));
    NCCLCHECK(ncclBcast(rbuf + (2 * blockcount)*typesize, blockcount, nccl_type, 3, nccl_local_comm, stream));
    NCCLCHECK(ncclBcast(rbuf + (3 * blockcount)*typesize, blockcount + count%blockcount, nccl_type, 4, nccl_local_comm, stream));
  }
}

void hybridAllReduce_nosplit(const void* sbuf, void* rbuf, size_t count, ncclDataType_t nccl_type, ncclRedOp_t nccl_op, ncclComm_t nccl_local_comm,
  cudaStream_t stream, void* rbuf_h, MPI_Datatype mpi_type, MPI_Op mpi_op, MPI_Comm local_comm, MPI_Comm node_comm, int lrank, int nsize)
{

  if (nsize > 1)
  {
    /* Reduce on node */
    NCCLCHECK(ncclReduce(sbuf, rbuf, count, nccl_type, nccl_op, 0, nccl_local_comm, stream));

    size_t blockcount = count;

    /* Intranode AllReduce using local rank 0*/
    if (lrank == 0)
    {
      int typesize;
      MPI_Type_size(mpi_type, &typesize);

      CUDACHECK(cudaMemcpyAsync(rbuf_h, rbuf,  blockcount*typesize, cudaMemcpyDeviceToHost, stream));
      cudaStreamSynchronize(stream);

      PUSH_RANGE("MPI_Allreduce", 0)
      MPI_Allreduce(MPI_IN_PLACE, rbuf_h, blockcount, mpi_type, mpi_op, node_comm);
      POP_RANGE

      CUDACHECK(cudaMemcpyAsync(rbuf, rbuf_h, blockcount*typesize, cudaMemcpyHostToDevice, stream));
      //cudaStreamSynchronize(stream);
    }

    MPI_Barrier(local_comm);
    //MPI_Barrier(node_comm);

    /* Bcast on node */
    NCCLCHECK(ncclBcast(rbuf, count, nccl_type, 0, nccl_local_comm, stream));
  }
  else
  {
    /* AllReduce on node */
    NCCLCHECK(ncclAllReduce(sbuf, rbuf, count, nccl_type, nccl_op, nccl_local_comm, stream));
  }
}

#endif

void hybridAllReduce_nosplit_cpu(const void* sbuf, void* rbuf, size_t count,
    MPI_Datatype mpi_type, MPI_Op mpi_op, MPI_Comm local_comm, MPI_Comm node_comm, int lrank, int nsize)
{

  if (nsize > 1)
  {
    /* Reduce on node */
    if (lrank == 0 and sbuf == rbuf)
      MPI_Reduce(MPI_IN_PLACE, rbuf, count, mpi_type, mpi_op, 0, local_comm);
    else
      MPI_Reduce(sbuf, rbuf, count, mpi_type, mpi_op, 0, local_comm);

    size_t blockcount = count;

    /* Intranode AllReduce */
    if (lrank == 0)
    {
      PUSH_RANGE("MPI_Allreduce", 0)
      MPI_Allreduce(MPI_IN_PLACE, rbuf, blockcount, mpi_type, mpi_op, node_comm);
      POP_RANGE
    }

    /* Bcast on node */
    MPI_Bcast(rbuf, blockcount, mpi_type, 0, local_comm);
  }
  else
  {
    /* AllReduce on node */
    if (sbuf == rbuf)
      MPI_Allreduce(MPI_IN_PLACE, rbuf, count, mpi_type, mpi_op, local_comm);
    else
      MPI_Allreduce(sbuf, rbuf, count, mpi_type, mpi_op, local_comm);
  }
}
