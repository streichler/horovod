#include "cuda_runtime.h"
#include "nccl.h"

#include "mpi.h"

#include "macros.h"
#include "hybrid_allreduce.h"

void hybridAllReduce(const float* sbuf, float* rbuf, size_t count, ncclComm_t nccl_local_comm,
  cudaStream_t stream, float* rbuf_h, MPI_Comm local_comm, MPI_Comm node_comm, int lrank, int nsize)
{
  /* AllReduce on node */
  NCCLCHECK(ncclAllReduce(sbuf, rbuf, count, ncclFloat, ncclSum, nccl_local_comm, stream));

  if (nsize > 1)
  {
    size_t blockcount = count/4;

    /* Internode AllReduce using local ranks 0,1,3,4*/
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

void hybridAllReduce_2split(const float* sbuf, float* rbuf, size_t count, ncclComm_t nccl_local_comm,
  cudaStream_t stream, float* rbuf_h, MPI_Comm local_comm, MPI_Comm node_comm, int lrank, int nsize)
{
  /* AllReduce on node */
  NCCLCHECK(ncclAllReduce(sbuf, rbuf, count, ncclFloat, ncclSum, nccl_local_comm, stream));

  if (nsize > 1)
  {
    size_t blockcount = count/2;

    /* Internode AllReduce using local rank 0,3*/
    if (lrank == 0  or lrank == 3)
    {
      size_t shift = 0;
      if (lrank == 3) shift = 1;

      CUDACHECK(cudaMemcpyAsync(rbuf_h, &rbuf[shift * blockcount], ((lrank == 3) ? blockcount + count%blockcount : blockcount)*sizeof(float),
                  cudaMemcpyDeviceToHost, stream));
      cudaStreamSynchronize(stream);

      PUSH_RANGE("MPI_Allreduce", 0)
      MPI_Allreduce(MPI_IN_PLACE, rbuf_h, ((lrank == 3) ? blockcount + count%blockcount : blockcount), MPI_FLOAT, MPI_SUM, node_comm);
      POP_RANGE

      CUDACHECK(cudaMemcpyAsync(&rbuf[shift * blockcount], rbuf_h, ((lrank == 3) ? blockcount + count%blockcount : blockcount)*sizeof(float),
                  cudaMemcpyHostToDevice, stream));
      //cudaStreamSynchronize(stream);
    }

    MPI_Barrier(local_comm);
    //MPI_Barrier(node_comm);

    /* Bcast on node */
    NCCLCHECK(ncclBcast(rbuf, blockcount, ncclFloat, 0, nccl_local_comm, stream));
    NCCLCHECK(ncclBcast(&rbuf[1 * blockcount], blockcount + count%blockcount, ncclFloat, 3, nccl_local_comm, stream));
  }
}

void hybridAllReduce_nosplit(const float* sbuf, float* rbuf, size_t count, ncclComm_t nccl_local_comm,
  cudaStream_t stream, float* rbuf_h, MPI_Comm local_comm, MPI_Comm node_comm, int lrank, int nsize)
{
  /* AllReduce on node */
  NCCLCHECK(ncclAllReduce(sbuf, rbuf, count, ncclFloat, ncclSum, nccl_local_comm, stream));

  if (nsize > 1)
  {
    size_t blockcount = count;

    /* Internode AllReduce using local rank 0*/
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
}
