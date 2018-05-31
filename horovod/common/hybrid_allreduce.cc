#if HAVE_NCCL
#include "cuda_runtime.h"
#include "nccl.h"
#endif

#include "mpi.h"

#include "macros.h"
#include "hybrid_allreduce.h"

#define ALIGN_FLOATS 256

#if HAVE_NCCL
void hybridAllReduce(const float* sbuf, float* rbuf, size_t count, ncclComm_t nccl_local_comm,
  cudaStream_t stream, float* rbuf_h, MPI_Comm local_comm, MPI_Comm node_comm, int lrank, int nsize)
{
  /* AllReduce on node */
  NCCLCHECK(ncclAllReduce(sbuf, rbuf, count, ncclFloat, ncclSum, nccl_local_comm, stream));

  if (nsize > 1)
  {
    size_t blockcount = count/ALIGN_FLOATS/4 * ALIGN_FLOATS; 

    // If buffer is too small to split, fallback to no split version
    if (blockcount == 0)
    {
       hybridAllReduce_nosplit(sbuf, rbuf, count, nccl_local_comm,
         stream, rbuf_h, local_comm, node_comm, lrank, nsize);
       return;
    }

    /* Intranode AllReduce using local ranks 0,1,3,4*/
    if (lrank == 0 or lrank == 1 or lrank == 3 or lrank == 4)
    {
      size_t shift = 0;
      if (lrank == 1) shift = 1;
      else if (lrank == 3) shift = 2;
      else if (lrank == 4) shift = 3;

      CUDACHECK(cudaMemcpyAsync(rbuf_h, &rbuf[shift * blockcount], ((lrank == 4) ? blockcount + count%blockcount : blockcount)*sizeof(float),
                  cudaMemcpyDeviceToHost, stream));
      cudaStreamSynchronize(stream);

      PUSH_RANGE("MPI_Allreduce", 0)
      MPI_Allreduce(MPI_IN_PLACE, rbuf_h, ((lrank == 4) ? blockcount + count%blockcount : blockcount), MPI_FLOAT, MPI_SUM, node_comm);
      POP_RANGE

      CUDACHECK(cudaMemcpyAsync(&rbuf[shift * blockcount], rbuf_h, ((lrank == 4) ? blockcount + count%blockcount : blockcount)*sizeof(float),
                  cudaMemcpyHostToDevice, stream));
      //cudaStreamSynchronize(stream);
    }

    MPI_Barrier(local_comm);
    //MPI_Barrier(node_comm);

    /* Bcast on node */
    NCCLCHECK(ncclBcast(rbuf, blockcount, ncclFloat, 0, nccl_local_comm, stream));
    NCCLCHECK(ncclBcast(&rbuf[1 * blockcount], blockcount, ncclFloat, 1, nccl_local_comm, stream));
    NCCLCHECK(ncclBcast(&rbuf[2 * blockcount], blockcount, ncclFloat, 3, nccl_local_comm, stream));
    NCCLCHECK(ncclBcast(&rbuf[3 * blockcount], blockcount + count%blockcount, ncclFloat, 4, nccl_local_comm, stream));
  }
}

void hybridAllReduce_nosplit(const float* sbuf, float* rbuf, size_t count, ncclComm_t nccl_local_comm,
  cudaStream_t stream, float* rbuf_h, MPI_Comm local_comm, MPI_Comm node_comm, int lrank, int nsize)
{

  if (nsize > 1)
  {
    /* Reduce on node */
    NCCLCHECK(ncclReduce(sbuf, rbuf, count, ncclFloat, ncclSum, 0, nccl_local_comm, stream));

    size_t blockcount = count;

    /* Intranode AllReduce using local rank 0*/
    if (lrank == 0)
    {

      CUDACHECK(cudaMemcpyAsync(rbuf_h, rbuf,  blockcount*sizeof(float), cudaMemcpyDeviceToHost, stream));
      cudaStreamSynchronize(stream);

      PUSH_RANGE("MPI_Allreduce", 0)
      MPI_Allreduce(MPI_IN_PLACE, rbuf_h, blockcount, MPI_FLOAT, MPI_SUM, node_comm);
      POP_RANGE

      CUDACHECK(cudaMemcpyAsync(rbuf, rbuf_h, blockcount*sizeof(float), cudaMemcpyHostToDevice, stream));
      //cudaStreamSynchronize(stream);
    }

    MPI_Barrier(local_comm);
    //MPI_Barrier(node_comm);

    /* Bcast on node */
    NCCLCHECK(ncclBcast(rbuf, count, ncclFloat, 0, nccl_local_comm, stream));
  }
  else
  {
    /* AllReduce on node */
    NCCLCHECK(ncclAllReduce(sbuf, rbuf, count, ncclFloat, ncclSum, nccl_local_comm, stream));
  }
}

#endif

void hybridAllReduce_nosplit_cpu(const float* sbuf, float* rbuf, size_t count,
    MPI_Comm local_comm, MPI_Comm node_comm, int lrank, int nsize)
{

  if (nsize > 1)
  {
    /* Reduce on node */
    if (lrank == 0 and sbuf == rbuf)
      MPI_Reduce(MPI_IN_PLACE, rbuf, count, MPI_FLOAT, MPI_SUM, 0, local_comm);
    else
      MPI_Reduce(sbuf, rbuf, count, MPI_FLOAT, MPI_SUM, 0, local_comm);

    size_t blockcount = count;

    /* Intranode AllReduce */
    if (lrank == 0)
    {
      PUSH_RANGE("MPI_Allreduce", 0)
      MPI_Allreduce(MPI_IN_PLACE, rbuf, blockcount, MPI_FLOAT, MPI_SUM, node_comm);
      POP_RANGE
    }

    /* Bcast on node */
    MPI_Bcast(rbuf, blockcount, MPI_FLOAT, 0, local_comm);
  }
  else
  {
    /* AllReduce on node */
    if (sbuf == rbuf)
      MPI_Allreduce(MPI_IN_PLACE, rbuf, count, MPI_FLOAT, MPI_SUM, local_comm);
    else
      MPI_Allreduce(sbuf, rbuf, count, MPI_FLOAT, MPI_SUM, local_comm);
  }
}
