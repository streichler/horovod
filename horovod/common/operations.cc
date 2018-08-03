// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2018 Uber Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

//#define USE_NVTX
//#define USE_HYBRID_ALLREDUCE

//#define NSOCKETS 2

#include <assert.h>
#include <atomic>
#include <cstring>
#include <queue>
#include <sstream>
#include <thread>
#include <unordered_map>

#if HAVE_CUDA
#include <cuda_runtime.h>
#endif

#if HAVE_NCCL
#include <nccl.h>
#endif

//#ifdef USE_HYBRID_ALLREDUCE
#include "hybrid_allreduce.h"
#include "fp16add.h"
//#endif

#define OMPI_SKIP_MPICXX
#include "hashes.h"
#include "mpi.h"
#include "mpi_message.h"
#include "operations.h"
#include "timeline.h"

#include "macros.h"

/*
 * Allreduce, Allgather and Broadcast Ops.
 *
 * This module implements MPI ops for allgather, allreduce and broadcast, which
 * do optimized gathers, reductions and broadcasts and can take advantage of
 * hardware-optimized communication libraries through the MPI implementation.
 *
 * The primary logic of the allreduce, allgather and broadcast are in MPI and
 * NCCL implementations. The background thread which facilitates MPI operations
 * is run in BackgroundThreadLoop(). The provided ops are:
 *      – HorovodAllreduce:
 *          Perform an allreduce on a Tensor, returning the sum
 *          across all MPI processes in the global communicator.
 *      – HorovodAllgather:
 *          Perform an allgather on a Tensor, returning the concatenation of
 *          the tensor on the first dimension across all MPI processes in the
 *          global communicator.
 *      - HorovodBroadcast:
 *          Perform a broadcast on a Tensor, broadcasting Tensor
 *          value from root rank to all other ranks.
 *
 * Additionally, this library provides C APIs to initialize Horovod and query
 * rank, local rank and world size.  These are used in Python directly through
 * ctypes.
 */

namespace horovod {
namespace common {

namespace {

// Table storing Tensors to be reduced, keyed by unique name.
// This table contains everything necessary to do the reduction.
typedef struct {
  // Name of the tensor.
  std::string tensor_name;
  // Operation context.
  std::shared_ptr<OpContext> context;
  // Input tensor.
  std::shared_ptr<Tensor> tensor;
  // Pre-allocated output tensor.
  std::shared_ptr<Tensor> output;
  // Root rank for broadcast operation.
  int root_rank;
  // Event indicating that data is ready.
  std::shared_ptr<ReadyEvent> ready_event;
  // GPU to do reduction on, or CPU_DEVICE_ID in case of CPU.
  int device;
  // A callback to call with the status.
  StatusCallback callback;
} TensorTableEntry;
typedef std::unordered_map<std::string, TensorTableEntry> TensorTable;

// Table for storing Tensor metadata on rank zero. This is used for error
// checking, stall checking and size calculations, as well as determining
// when a reduction is ready to be done (when all nodes are ready to do it).
typedef std::unordered_map<
    std::string,
    std::tuple<std::vector<MPIRequest>, std::chrono::steady_clock::time_point>>
    MessageTable;

// The global state required for the MPI ops.
//
// MPI is a library that stores a lot of global per-program state and often
// requires running on a single thread. As a result, we have to have a single
// background thread responsible for all MPI operations, and communicate with
// that background thread through global state.
struct HorovodGlobalState {
  // An atomic boolean which is set to true when background thread is started.
  // This ensures that only one background thread is spawned.
  std::atomic_flag initialize_flag = ATOMIC_FLAG_INIT;

  // A mutex that needs to be used whenever MPI operations are done.
  std::mutex mutex;

  // Tensors waiting to be allreduced or allgathered.
  TensorTable tensor_table;

  // Queue of MPI requests waiting to be sent to the coordinator node.
  std::deque<MPIRequest> message_queue;

  // Background thread running MPI communication.
  std::thread background_thread;

  // Whether the background thread should shutdown.
  bool shut_down = false;

  // Only exists on the coordinator node (rank zero). Maintains a count of
  // how many nodes are ready to allreduce every tensor (keyed by tensor
  // name) and time point when tensor started allreduce op.
  std::unique_ptr<MessageTable> message_table;

  // Time point when coordinator last checked for stalled tensors.
  std::chrono::steady_clock::time_point last_stall_check;

  // Timeline writer.
  Timeline timeline;

  // Threshold for Tensor Fusion.  All tensors that occupy memory beyond this
  // threshold will be fused.
  int64_t tensor_fusion_threshold = 64 * 1024 * 1024;

  // Memory buffers for Tensor Fusion.  They are keyed off device ID and
  // framework, and all are allocated tensor_fusion_threshold bytes if
  // initialized.
  std::unordered_map<std::tuple<int, Framework>,
                     std::shared_ptr<PersistentBuffer>>
      tensor_fusion_buffers;
//#ifdef USE_HYBRID_ALLREDUCE
  std::unordered_map<std::tuple<int, Framework>,
                     void*>
      tensor_fusion_buffers_h;
//#endif

  // Whether MPI_Init has been completed on the background thread.
  bool initialization_done = false;

  // The MPI rank, local rank, size, local size and flag indicating whether MPI
  // multi-threading is supported.
  int rank = 0;
  int local_rank = 0;
  int size = 1;
  int local_size = 1;
  bool mpi_threads_supported = false;

  // Custom op for fp16 reduction
  MPI_Op Float16SumOp;

//#ifdef USE_HYBRID_ALLREDUCE
  int node_size = 1;
  int node_rank = 0;
  MPI_Comm local_comm;
  MPI_Comm node_comm;

  int local_size_socket = 1;
  int local_rank_socket = 0;
  int node_size_socket = 1;
  int node_rank_socket = 0;
  MPI_Comm local_comm_socket;
  MPI_Comm node_comm_socket;
//#endif

  std::array<char, MPI_MAX_PROCESSOR_NAME> hostname;
  std::vector<std::array<char, MPI_MAX_PROCESSOR_NAME>> hostlist;
  std::vector<int> local_rank_list;

// The CUDA stream used for data transfers and within-allreduce operations.
// A naive implementation would use the TensorFlow StreamExecutor CUDA
// stream. However, the allreduce and allgather require doing memory copies
// and kernel executions (for accumulation of values on the GPU). However,
// the subsequent operations must wait for those operations to complete,
// otherwise MPI (which uses its own stream internally) will begin the data
// transfers before the CUDA calls are complete. In order to wait for those
// CUDA operations, if we were using the TensorFlow stream, we would have to
// synchronize that stream; however, other TensorFlow threads may be
// submitting more work to that stream, so synchronizing on it can cause the
// allreduce to be delayed, waiting for compute totally unrelated to it in
// other parts of the graph. Overlaying memory transfers and compute during
// backpropagation is crucial for good performance, so we cannot use the
// TensorFlow stream, and must use our own stream.
#if HAVE_CUDA
  std::unordered_map<int, cudaStream_t> streams;
#endif
#if HAVE_NCCL
  std::unordered_map<std::vector<int32_t>, ncclComm_t> nccl_comms;
#endif

// We reuse CUDA events as it appears that their creation carries non-zero cost.
// Event management code is only used in NCCL path.
#if HAVE_NCCL
  std::unordered_map<int, std::queue<cudaEvent_t>> cuda_events;
  std::mutex cuda_events_mutex;
#endif

  ~HorovodGlobalState() {
    // Make sure that the destructor of the background thread is safe to
    // call. If a thread is still joinable (not detached or complete) its
    // destructor cannot be called.
    if (background_thread.joinable()) {
      shut_down = true;
      background_thread.join();
    }
  }
};

// All the Horovod state that must be stored globally per-process.
static HorovodGlobalState horovod_global;

// For clarify in argument lists.
#define RANK_ZERO 0

// A tag used for all coordinator messaging.
#define TAG_NOTIFY 1

// Stall-check warning time
#define STALL_WARNING_TIME std::chrono::seconds(60)

// Store the MPIRequest for a name, and return whether the total count of
// MPIRequests for that tensor is now equal to the MPI size (and thus we are
// ready to reduce the tensor).
bool IncrementTensorCount(std::unique_ptr<MessageTable>& message_table,
                          MPIRequest msg, int mpi_size) {
  auto name = msg.tensor_name();
  auto& timeline = horovod_global.timeline;
  auto table_iter = message_table->find(name);
  if (table_iter == message_table->end()) {
    std::vector<MPIRequest> messages = {msg};
    auto now = std::chrono::steady_clock::now();
    message_table->emplace(name, std::make_tuple(std::move(messages), now));
    table_iter = message_table->find(name);
    timeline.NegotiateStart(name, msg.request_type());
  } else {
    std::vector<MPIRequest>& messages = std::get<0>(table_iter->second);
    messages.push_back(msg);
  }

  timeline.NegotiateRankReady(name, msg.request_rank());

  std::vector<MPIRequest>& messages = std::get<0>(table_iter->second);
  int count = (int)messages.size();
  bool ready_to_reduce = count == mpi_size;
  if (ready_to_reduce) {
    timeline.NegotiateEnd(name);
  }
  return ready_to_reduce;
}

// Once a tensor is ready to be reduced, the coordinator sends an MPIResponse
// instructing all ranks to start the reduction to all ranks. The MPIResponse
// also contains error messages in case the submitted MPIRequests were not
// valid (for example, contained mismatched shapes or types).
//
// Constructing the MPIResponse, thus, requires a whole lot of error checking.
MPIResponse ConstructMPIResponse(std::unique_ptr<MessageTable>& message_table,
                                 std::string name) {
  bool error = false;
  auto it = message_table->find(name);
  assert(it != message_table->end());

  std::vector<MPIRequest>& requests = std::get<0>(it->second);
  assert(requests.size() > 0);

  std::ostringstream error_message_stream;

  // Check that all data types of tensors being reduced, gathered or broadcasted
  // are identical.
  auto data_type = requests[0].tensor_type();
  for (unsigned int i = 1; i < requests.size(); i++) {
    auto request_type = requests[i].tensor_type();
    if (data_type != request_type) {
      error = true;
      error_message_stream << "Mismatched data types: One rank had type "
                           << MPIDataType_Name(data_type)
                           << ", but another rank had type "
                           << MPIDataType_Name(request_type) << ".";
      break;
    }
  }

  // Check that all requested operations are the same
  auto message_type = requests[0].request_type();
  for (unsigned int i = 1; i < requests.size(); i++) {
    if (error) {
      break;
    }

    auto request_type = requests[i].request_type();
    if (message_type != request_type) {
      error = true;
      error_message_stream << "Mismatched MPI operations: One rank did an "
                           << MPIRequest::RequestType_Name(message_type)
                           << ", but another rank did an "
                           << MPIRequest::RequestType_Name(request_type) << ".";
      break;
    }
  }

  // If we are doing an allreduce or broadcast, check that all tensor shapes are
  // identical.
  if (message_type == MPIRequest::ALLREDUCE ||
      message_type == MPIRequest::BROADCAST) {
    TensorShape tensor_shape;
    for (auto it = requests[0].tensor_shape().begin();
         it != requests[0].tensor_shape().end(); it++) {
      tensor_shape.AddDim(*it);
    }
    for (unsigned int i = 1; i < requests.size(); i++) {
      if (error) {
        break;
      }

      TensorShape request_shape;
      for (auto it = requests[i].tensor_shape().begin();
           it != requests[i].tensor_shape().end(); it++) {
        request_shape.AddDim(*it);
      }
      if (tensor_shape != request_shape) {
        error = true;
        error_message_stream
            << "Mismatched " << MPIRequest::RequestType_Name(message_type)
            << " tensor shapes: One rank sent a tensor of shape "
            << tensor_shape.DebugString()
            << ", but another rank sent a tensor of shape "
            << request_shape.DebugString() << ".";
        break;
      }
    }
  }

  // If we are doing an allgather, make sure all but the first dimension are
  // the same. The first dimension may be different and the output tensor is
  // the sum of the first dimension. Collect the sizes by rank.
  std::vector<int64_t> tensor_sizes(requests.size());
  if (message_type == MPIRequest::ALLGATHER) {
    TensorShape tensor_shape;
    for (auto it = requests[0].tensor_shape().begin();
         it != requests[0].tensor_shape().end(); it++) {
      tensor_shape.AddDim(*it);
    }

    if (tensor_shape.dims() == 0) {
      error = true;
      error_message_stream << "Rank zero tried to "
                           << MPIRequest::RequestType_Name(message_type)
                           << " a rank-zero tensor.";
    } else {
#ifdef DEVICES_LIST_IS_BROKEN_ANYWAY
      tensor_sizes[requests[0].request_rank()] = tensor_shape.dim_size(0);
#else
      error = true;
      error_message_stream << "Non-scalar allgathers not currently supported.";
#endif
    }

    for (unsigned int i = 1; i < requests.size(); i++) {
      if (error) {
        break;
      }

      TensorShape request_shape;
      for (auto it = requests[i].tensor_shape().begin();
           it != requests[i].tensor_shape().end(); it++) {
        request_shape.AddDim(*it);
      }
      if (tensor_shape.dims() != request_shape.dims()) {
        error = true;
        error_message_stream
            << "Mismatched " << MPIRequest::RequestType_Name(message_type)
            << " tensor shapes: One rank sent a tensor of rank "
            << tensor_shape.dims()
            << ", but another rank sent a tensor of rank "
            << request_shape.dims() << ".";
        break;
      }

      bool dim_mismatch = false;
      for (int dim = 1; dim < tensor_shape.dims(); dim++) {
        if (tensor_shape.dim_size(dim) != request_shape.dim_size(dim)) {
          error = true;
          error_message_stream
              << "Mismatched " << MPIRequest::RequestType_Name(message_type)
              << " tensor shapes: One rank sent a tensor with dimension " << dim
              << " equal to " << tensor_shape.dim_size(dim)
              << ", but another rank sent a tensor with dimension " << dim
              << " equal to " << request_shape.dim_size(dim) << ".";
          dim_mismatch = true;
          break;
        }
      }
      if (dim_mismatch) {
        break;
      }

      tensor_sizes[requests[i].request_rank()] = request_shape.dim_size(0);
    }
  }

  // If we are doing a broadcast, check that all root ranks are identical.
  if (message_type == MPIRequest::BROADCAST) {
    int first_root_rank = requests[0].root_rank();
    for (unsigned int i = 1; i < requests.size(); i++) {
      if (error) {
        break;
      }

      int this_root_rank = requests[i].root_rank();
      if (first_root_rank != this_root_rank) {
        error = true;
        error_message_stream
            << "Mismatched " << MPIRequest::RequestType_Name(message_type)
            << " root ranks: One rank specified root rank " << first_root_rank
            << ", but another rank specified root rank " << this_root_rank
            << ".";
        break;
      }
    }
  }

  bool first_device_is_cpu = requests[0].device() == CPU_DEVICE_ID;
  for (unsigned int i = 1; i < requests.size(); i++) {
    if (error) {
      break;
    }

    bool this_device_is_cpu = requests[i].device() == CPU_DEVICE_ID;
    if (first_device_is_cpu != this_device_is_cpu) {
      error = true;
      error_message_stream
          << "Mismatched " << MPIRequest::RequestType_Name(message_type)
          << " CPU/GPU device selection: One rank specified device "
          << (first_device_is_cpu ? "CPU" : "GPU")
          << ", but another rank specified device "
          << (this_device_is_cpu ? "CPU" : "GPU") << ".";
      break;
    }
  }
#ifdef DEVICES_LIST_IS_BROKEN_ANYWAY
  std::vector<int32_t> devices(requests.size());
  for (auto it = requests.begin(); it != requests.end(); it++) {
    devices[it->request_rank()] = it->device();
  }
#endif

  MPIResponse response;
  response.add_tensor_names(name);
  if (error) {
    std::string error_message = error_message_stream.str();
    response.set_response_type(MPIResponse::ERROR);
    response.set_error_message(error_message);
  } else if (message_type == MPIRequest::ALLGATHER) {
    response.set_response_type(MPIResponse::ALLGATHER);
    for (auto dim : tensor_sizes) {
      response.add_tensor_sizes(dim);
    }
  } else if (message_type == MPIRequest::ALLREDUCE) {
    response.set_response_type(MPIResponse::ALLREDUCE);
  } else if (message_type == MPIRequest::BROADCAST) {
    response.set_response_type(MPIResponse::BROADCAST);
  }
#ifdef DEVICES_LIST_IS_BROKEN_ANYWAY
  response.set_devices(devices);
#endif

  // Clear all queued up requests for this name. They are now taken care of
  // by the constructed MPI response.
  message_table->erase(it);

  return response;
}

MPI_Datatype GetMPIDataType(const std::shared_ptr<Tensor> tensor) {
  switch (tensor->dtype()) {
  case HOROVOD_UINT8:
    return MPI_UINT8_T;
  case HOROVOD_INT8:
    return MPI_INT8_T;
  case HOROVOD_UINT16:
    return MPI_UINT16_T;
  case HOROVOD_INT16:
    return MPI_INT16_T;
  case HOROVOD_INT32:
    return MPI_INT32_T;
  case HOROVOD_INT64:
    return MPI_INT64_T;
  case HOROVOD_FLOAT16:
    return MPI_UINT16_T;
  case HOROVOD_FLOAT32:
    return MPI_FLOAT;
  case HOROVOD_FLOAT64:
    return MPI_DOUBLE;
  case HOROVOD_BOOL:
    return MPI_C_BOOL;
  default:
    throw std::logic_error("Type " + MPIDataType_Name(tensor->dtype()) +
                           " is not supported in MPI mode.");
  }
}

MPI_Op GetMPISumOp(const std::shared_ptr<Tensor> tensor) {
  if (tensor->dtype() == HOROVOD_FLOAT16) {
    return horovod_global.Float16SumOp;
  } else {
    return MPI_SUM;
  }
}

#if HAVE_NCCL
ncclDataType_t GetNCCLDataType(const std::shared_ptr<Tensor> tensor) {
  switch (tensor->dtype()) {
  case HOROVOD_INT32:
    return ncclInt32;
  case HOROVOD_INT64:
    return ncclInt64;
  case HOROVOD_FLOAT16:
    return ncclFloat16;
  case HOROVOD_FLOAT32:
    return ncclFloat32;
  case HOROVOD_FLOAT64:
    return ncclFloat64;
  default:
    throw std::logic_error("Type " + MPIDataType_Name(tensor->dtype()) +
                           " is not supported in NCCL mode.");
  }
}
#endif

// Custom reduction op for fp16 MPI_Allreduce.
void fp16_sum_reduce(void* in, void* inout, int* len, MPI_Datatype *dtype) {
  // Not defined. Just a NO OP for now.
  cpu_add((HALF *) in, (HALF *) inout, (size_t) *len);
   
}

#define MPI_CHECK(entries, op_name, op)                                        \
  {                                                                            \
    auto mpi_result = (op);                                                    \
    if (mpi_result != MPI_SUCCESS) {                                           \
      for (auto it = entries.begin(); it != entries.end(); it++) {             \
        timeline.End(it->tensor_name, nullptr);                                \
        it->callback(Status::UnknownError(                                     \
            std::string(op_name) + " failed, see MPI output for details."));   \
      }                                                                        \
      return;                                                                  \
    }                                                                          \
  }

#define CUDA_CHECK(entries, op_name, op)                                       \
  {                                                                            \
    auto cuda_result = (op);                                                   \
    if (cuda_result != cudaSuccess) {                                          \
      for (auto it = entries.begin(); it != entries.end(); it++) {             \
        timeline.End(it->tensor_name, nullptr);                                \
        it->callback(Status::UnknownError(std::string(op_name) + " failed: " + \
                                          cudaGetErrorString(cuda_result)));   \
      }                                                                        \
      return;                                                                  \
    }                                                                          \
  }

#define NCCL_CHECK(entries, op_name, op)                                       \
  {                                                                            \
    auto nccl_result = (op);                                                   \
    if (nccl_result != ncclSuccess) {                                          \
      for (auto it = entries.begin(); it != entries.end(); it++) {             \
        timeline.End(it->tensor_name, nullptr);                                \
        it->callback(Status::UnknownError(std::string(op_name) + " failed: " + \
                                          ncclGetErrorString(nccl_result)));   \
      }                                                                        \
      return;                                                                  \
    }                                                                          \
  }

// This event management code is only used in NCCL.
#ifdef HAVE_NCCL
cudaError_t GetCudaEvent(cudaEvent_t* event) {
  int device;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

  auto& mutex = horovod_global.cuda_events_mutex;
  auto& queue = horovod_global.cuda_events[device];
  {
    std::lock_guard<std::mutex> guard(mutex);
    if (!queue.empty()) {
      *event = queue.front();
      queue.pop();
      return cudaSuccess;
    }
  }

  return cudaEventCreateWithFlags(event, cudaEventBlockingSync |
                                             cudaEventDisableTiming);
}

cudaError_t ReleaseCudaEvent(cudaEvent_t event) {
  int device;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

  auto& mutex = horovod_global.cuda_events_mutex;
  auto& queue = horovod_global.cuda_events[device];
  {
    std::lock_guard<std::mutex> guard(mutex);
    queue.push(event);
  }

  return cudaSuccess;
}

#define RECORD_EVENT(entries, event, stream)                                   \
  CUDA_CHECK(entries, "GetCudaEvent", GetCudaEvent(&event))                    \
  CUDA_CHECK(entries, "cudaEventRecord", cudaEventRecord(event, stream))

#define RELEASE_EVENT(entries, event)                                          \
  CUDA_CHECK(entries, "ReleaseCudaEvent", ReleaseCudaEvent(event))
#endif

#define ACTIVITY_START_ALL(entries, timeline, activity)                        \
  {                                                                            \
    for (auto it = entries.begin(); it != entries.end(); it++) {               \
      timeline.ActivityStart(it->tensor_name, activity);                       \
    }                                                                          \
  }

#define ACTIVITY_END_ALL(entries, timeline)                                    \
  {                                                                            \
    for (auto it = entries.begin(); it != entries.end(); it++) {               \
      timeline.ActivityEnd(it->tensor_name);                                   \
    }                                                                          \
  }

// Process an MPIResponse by doing a reduction, a gather, a broadcast, or
// raising an error.
void PerformOperation(TensorTable& tensor_table, MPIResponse response) {
  std::vector<TensorTableEntry> entries;
  {
    // Lock on the tensor table.
    std::lock_guard<std::mutex> guard(horovod_global.mutex);

    for (auto it = response.tensor_names().begin();
         it != response.tensor_names().end(); it++) {
      // We should never fail at finding this key in the tensor table.
      auto name = *it;
      auto iter = tensor_table.find(name);
      assert(iter != tensor_table.end());

      assert(response.response_type() == MPIResponse::ALLREDUCE ||
             response.response_type() == MPIResponse::ALLGATHER ||
             response.response_type() == MPIResponse::BROADCAST ||
             response.response_type() == MPIResponse::ERROR);

      entries.push_back(iter->second);

      // Clear the tensor table of this tensor and its callbacks; the rest of
      // this function takes care of it.
      tensor_table.erase(iter);
    }
  }

  auto& timeline = horovod_global.timeline;
  for (auto it = entries.begin(); it != entries.end(); it++) {
    timeline.Start(it->tensor_name, response.response_type());
  }

  if (entries.size() > 1) {
    auto first_entry = entries[0];
    // Note: it is OK for different entries to come from different frameworks
    // since buffer allocated here is guaranteed to survive at least till the
    // end of this operation.
    auto& buffer = horovod_global.tensor_fusion_buffers[std::make_tuple(
        first_entry.device, first_entry.context->framework())];
    if (buffer == nullptr) {
      ACTIVITY_START_ALL(entries, timeline, "INIT_FUSION_BUFFER")

      // Lazily allocate persistent buffer for Tensor Fusion and keep it
      // forever per device.
      Status status = first_entry.context->AllocatePersistent(
          horovod_global.tensor_fusion_threshold, &buffer);
      if (!status.ok()) {
        for (auto it = entries.begin(); it != entries.end(); it++) {
          timeline.End(it->tensor_name, nullptr);
          it->callback(status);
        }
        return;
      }

      ACTIVITY_END_ALL(entries, timeline)
    }
  }

  auto horovod_use_hybrid_allreduce = std::getenv("HOROVOD_USE_HYBRID_ALLREDUCE");
//#ifdef USE_HYBRID_ALLREDUCE
  if (horovod_use_hybrid_allreduce != nullptr) {
#ifdef HAVE_CUDA
    // Allocate pinned host fusion buffer, regardless of entries.size()
    if (entries.size() > 0) {
      auto first_entry = entries[0];
      bool on_gpu = first_entry.device != CPU_DEVICE_ID;
      if (on_gpu)
      {
        auto& buffer_h = horovod_global.tensor_fusion_buffers_h[std::make_tuple(
            first_entry.device, first_entry.context->framework())];
        if (buffer_h == nullptr) {
          if (horovod_global.rank == 0) printf("Allocating pinned buffers on host...\n");
          CUDACHECK(cudaMallocHost(&buffer_h, horovod_global.tensor_fusion_threshold));
        }
      }
    }
  }
#endif
//#endif

  // On GPU data readiness is signalled by ready_event.
  PUSH_RANGE("horovod: WAIT FOR DATA", 3)
  std::vector<TensorTableEntry> waiting_tensors;
  for (auto it = entries.begin(); it != entries.end(); it++) {
    if (it->ready_event != nullptr) {
      timeline.ActivityStart(it->tensor_name, "WAIT_FOR_DATA");
      waiting_tensors.push_back(*it);
    }
  }
  while (!waiting_tensors.empty()) {
    for (auto it = waiting_tensors.begin(); it != waiting_tensors.end();) {
      if (it->ready_event->Ready()) {
        timeline.ActivityEnd(it->tensor_name);
        timeline.ActivityStart(it->tensor_name, "WAIT_FOR_OTHER_TENSOR_DATA");
        it = waiting_tensors.erase(it);
      } else {
        it++;
      }
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(100));
  }
  for (auto it = entries.begin(); it != entries.end(); it++) {
    if (it->ready_event != nullptr) {
      timeline.ActivityEnd(it->tensor_name);
    }
  }
  POP_RANGE

  Status status;
  if (response.response_type() == MPIResponse::ALLGATHER) {
    assert(entries.size() == 1);
    auto e = entries[0];

    // Copy tensor sizes from the MPI response into a vector of int64_t
    // and compute total size.  This is size of first dimension.
    std::vector<int64_t> tensor_sizes;
    int64_t total_dimension_size = 0;
    for (auto it = response.tensor_sizes().begin();
         it != response.tensor_sizes().end(); it++) {
      tensor_sizes.push_back(*it);
      total_dimension_size += *it;
    }

    // Every tensor participating in Allgather operation may have different
    // first dimension size, but the rest of dimensions are same for all
    // tensors.  Here we get shape of tensor sliced by first dimension.
    TensorShape single_slice_shape;
    for (int i = 1; i < e.tensor->shape().dims(); i++) {
      single_slice_shape.AddDim(e.tensor->shape().dim_size(i));
    }

    // Allgather output will have shape of:
    // (sum of first dimension of every tensor) x (tensor slice shape).
    TensorShape output_shape;
    output_shape.AddDim((int64_t)total_dimension_size);
    output_shape.AppendShape(single_slice_shape);

    ACTIVITY_START_ALL(entries, timeline, "ALLOCATE_OUTPUT")
    status = e.context->AllocateOutput(output_shape, &e.output);
    if (!status.ok()) {
      timeline.End(e.tensor_name, nullptr);
      e.callback(status);
      return;
    }
    ACTIVITY_END_ALL(entries, timeline)

    // Tensors may have different first dimension, so we need to use
    // MPI_Allgatherv API that supports gathering arrays of different length.
    ACTIVITY_START_ALL(entries, timeline, "MPI_ALLGATHER")
    int* recvcounts = new int[tensor_sizes.size()];
    int* displcmnts = new int[tensor_sizes.size()];
    for (unsigned int i = 0; i < tensor_sizes.size(); i++) {
      recvcounts[i] =
          (int)(single_slice_shape.num_elements() * tensor_sizes[i]);
      if (i == 0) {
        displcmnts[i] = 0;
      } else {
        displcmnts[i] = recvcounts[i - 1] + displcmnts[i - 1];
      }
    }
    auto result = MPI_Allgatherv(
        (const void*)e.tensor->data(), (int)e.tensor->shape().num_elements(),
        GetMPIDataType(e.tensor), (void*)e.output->data(), recvcounts,
        displcmnts, GetMPIDataType(e.tensor), MPI_COMM_WORLD);
    delete[] recvcounts;
    delete[] displcmnts;
    MPI_CHECK(entries, "MPI_Allgatherv", result)
    ACTIVITY_END_ALL(entries, timeline)

    timeline.End(e.tensor_name, e.output);
    e.callback(Status::OK());

  } else if (response.response_type() == MPIResponse::ALLREDUCE) {
    auto first_entry = entries[0];
#if HAVE_CUDA
    bool on_gpu = first_entry.device != CPU_DEVICE_ID;
    if (on_gpu) {
      CUDA_CHECK(entries, "cudaSetDevice", cudaSetDevice(first_entry.device))

      // Ensure stream is in the map before executing reduction.
      cudaStream_t& stream = horovod_global.streams[first_entry.device];
      if (stream == nullptr) {
        auto horovod_use_priority = std::getenv("HOROVOD_USE_PRIORITY");

	if (horovod_use_priority != nullptr) {
          if (horovod_global.rank == 0) printf("Using priority stream for NCCL...\n");

          int greatestPriority;
          CUDACHECK(cudaDeviceGetStreamPriorityRange(NULL, &greatestPriority));
          CUDACHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPriority));
        } else {
          CUDA_CHECK(entries, "cudaStreamCreate", cudaStreamCreate(&stream))
        }

      }
    }
#endif

#if HOROVOD_GPU_ALLREDUCE == 'N' // 'N' stands for NCCL
    if (on_gpu) {
      auto stream = horovod_global.streams[first_entry.device];

      // Ensure NCCL communicator is in the map before executing reduction.
      ncclComm_t& nccl_comm = horovod_global.nccl_comms[response.devices()];
      if (nccl_comm == nullptr) {
        ACTIVITY_START_ALL(entries, timeline, "INIT_NCCL")

        ncclUniqueId nccl_id;
//#ifndef USE_HYBRID_ALLREDUCE
        if (horovod_use_hybrid_allreduce == nullptr) {
          if (horovod_global.rank == 0) printf("Using NCCL allreduce, creating global NCCL communicator...\n");

          if (horovod_global.rank == 0) {
            NCCL_CHECK(entries, "ncclGetUniqueId", ncclGetUniqueId(&nccl_id))
          }

          MPI_CHECK(entries, "MPI_Bcast",
                    MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0,
                              MPI_COMM_WORLD));

          ncclComm_t new_nccl_comm;
          NCCL_CHECK(entries, "ncclCommInitRank",
                     ncclCommInitRank(&new_nccl_comm, horovod_global.size,
                                      nccl_id, horovod_global.rank))
          nccl_comm = new_nccl_comm;
//#else
        } else {
          if (horovod_global.rank == 0) printf("Using HYBRID allreduce, creating node local NCCL communicators...\n");

          // Replacing global nccl communicator with node local communicator
          if (horovod_global.local_rank == 0) {
            NCCL_CHECK(entries, "ncclGetUniqueId", ncclGetUniqueId(&nccl_id))
          }

          MPI_CHECK(entries, "MPI_Bcast",
                    MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0,
                              horovod_global.local_comm));

          ncclComm_t new_nccl_comm;
          NCCL_CHECK(entries, "ncclCommInitRank",
                     ncclCommInitRank(&new_nccl_comm, horovod_global.local_size, nccl_id,
                                      horovod_global.local_rank))
          nccl_comm = new_nccl_comm;
        }
//#endif

        // Barrier helps NCCL to synchronize after initialization and avoid
        // deadlock that we've been seeing without it.
        MPI_CHECK(entries, "MPI_Barrier", MPI_Barrier(MPI_COMM_WORLD));

        ACTIVITY_END_ALL(entries, timeline)
      }

      ACTIVITY_START_ALL(entries, timeline, "SCHEDULE")

      cudaEvent_t queue_end_event = nullptr;
      if (timeline.Initialized()) {
        RECORD_EVENT(entries, queue_end_event, stream);
      }

      cudaEvent_t after_memcpy_in_event = nullptr;
      cudaEvent_t after_reduce_event = nullptr;
      cudaEvent_t after_memcpy_out_event = nullptr;
      if (entries.size() > 1) {
        // Access the fusion buffer.
        auto& buffer = horovod_global.tensor_fusion_buffers[std::make_tuple(
            first_entry.device, first_entry.context->framework())];
        auto buffer_data = buffer->AccessData(first_entry.context);

        // Copy memory into the fusion buffer.
        int64_t offset = 0;
        for (auto it = entries.begin(); it != entries.end(); it++) {
          CUDA_CHECK(entries, "cudaMemcpyAsync",
                     cudaMemcpyAsync((void*)(buffer_data + offset),
                                     (const void*)it->tensor->data(),
                                     (size_t)it->tensor->size(),
                                     cudaMemcpyDeviceToDevice, stream))
          offset += it->tensor->size();
        }
        if (timeline.Initialized()) {
          RECORD_EVENT(entries, after_memcpy_in_event, stream)
        }

        // Perform the reduction on the fusion buffer.
        int64_t num_elements = 0;
        for (auto it = entries.begin(); it != entries.end(); it++) {
          num_elements += it->tensor->shape().num_elements();
        }
//#ifndef USE_HYBRID_ALLREDUCE
        if (horovod_use_hybrid_allreduce == nullptr) {
          NCCL_CHECK(entries, "ncclAllReduce",
                     ncclAllReduce((const void*)buffer_data, (void*)buffer_data,
                                   (size_t)num_elements,
                                   GetNCCLDataType(first_entry.tensor), ncclSum,
                                   nccl_comm, stream))
//#else
        } else {
          // Access host fusion buffer
          auto buffer_data_h = horovod_global.tensor_fusion_buffers_h[std::make_tuple(
              first_entry.device, first_entry.context->framework())];

          hybridAllReduce((const void*)buffer_data, (void*)buffer_data, num_elements, 
                          GetNCCLDataType(first_entry.tensor), ncclSum, nccl_comm,
                          stream, (void*)buffer_data_h, GetMPIDataType(first_entry.tensor), 
                          GetMPISumOp(first_entry.tensor), horovod_global.local_comm, horovod_global.node_comm,
                          horovod_global.local_rank, horovod_global.node_size);
        }
//#endif
        if (timeline.Initialized()) {
          RECORD_EVENT(entries, after_reduce_event, stream)
        }

        // Copy memory out of the fusion buffer.
        offset = 0;
        for (auto it = entries.begin(); it != entries.end(); it++) {
          CUDA_CHECK(entries, "cudaMemcpyAsync",
                     cudaMemcpyAsync((void*)it->output->data(),
                                     (const void*)(buffer_data + offset),
                                     (size_t)it->tensor->size(),
                                     cudaMemcpyDeviceToDevice, stream))
          offset += it->tensor->size();
        }
        if (timeline.Initialized()) {
          RECORD_EVENT(entries, after_memcpy_out_event, stream)
        }
      } else {
        auto e = first_entry;
//#ifndef USE_HYBRID_ALLREDUCE
        if (horovod_use_hybrid_allreduce == nullptr) {
          NCCL_CHECK(entries, "ncclAllReduce",
                     ncclAllReduce((const void*)e.tensor->data(),
                                   (void*)e.output->data(),
                                   (size_t)e.tensor->shape().num_elements(),
                                   GetNCCLDataType(first_entry.tensor), ncclSum,
                                   nccl_comm, stream))
//#else
        } else {
          auto buffer_data_h = horovod_global.tensor_fusion_buffers_h[std::make_tuple(
              e.device, e.context->framework())];

          hybridAllReduce((const void*)e.tensor->data(), (void*)e.output->data(),
                          (size_t)e.tensor->shape().num_elements(), 
                          GetNCCLDataType(first_entry.tensor), ncclSum, nccl_comm, stream, 
                          (void*)buffer_data_h, GetMPIDataType(first_entry.tensor), 
                          GetMPISumOp(first_entry.tensor), horovod_global.local_comm, horovod_global.node_comm, 
                          horovod_global.local_rank, horovod_global.node_size);
        }
//#endif
        if (timeline.Initialized()) {
          RECORD_EVENT(entries, after_reduce_event, stream)
        }
      }

      ACTIVITY_END_ALL(entries, timeline)
      ACTIVITY_START_ALL(entries, timeline, "QUEUE")

      // Use completion marker via event because it's faster than
      // blocking cudaStreamSynchronize() in this thread.
      cudaEvent_t done_event;
      RECORD_EVENT(entries, done_event, stream)

      // TODO: use thread pool or single thread for callbacks
      std::thread finalizer_thread([entries, first_entry, done_event,
                                    queue_end_event, after_memcpy_in_event,
                                    after_reduce_event, after_memcpy_out_event,
                                    response, &timeline] {
        PUSH_RANGE("horovod: NCCL ALLREDUCE", 4)

        CUDA_CHECK(entries, "cudaSetDevice", cudaSetDevice(first_entry.device))
        if (queue_end_event != nullptr) {
          CUDA_CHECK(entries, "cudaEventSynchronize",
                     cudaEventSynchronize(queue_end_event))
          // All the work scheduled on NCCL stream before this allreduce
          // is done at this point, end queueing activity.
          ACTIVITY_END_ALL(entries, timeline)
          RELEASE_EVENT(entries, queue_end_event);
        }

        if (after_memcpy_in_event != nullptr) {
          ACTIVITY_START_ALL(entries, timeline, "MEMCPY_IN_FUSION_BUFFER")
          CUDA_CHECK(entries, "cudaEventSynchronize",
                     cudaEventSynchronize(after_memcpy_in_event))
          // The memcpy into the fusion buffer is done after this point has been
          // reached.
          ACTIVITY_END_ALL(entries, timeline)
          RELEASE_EVENT(entries, after_memcpy_in_event);
        }

        if (after_reduce_event != nullptr) {
          ACTIVITY_START_ALL(entries, timeline, "NCCL_ALLREDUCE")
          CUDA_CHECK(entries, "cudaEventSynchronize",
                     cudaEventSynchronize(after_reduce_event))
          // The allreduce is done after this point has been reached.
          ACTIVITY_END_ALL(entries, timeline)
          RELEASE_EVENT(entries, after_reduce_event);
        }

        if (after_memcpy_out_event != nullptr) {
          ACTIVITY_START_ALL(entries, timeline, "MEMCPY_OUT_FUSION_BUFFER")
          CUDA_CHECK(entries, "cudaEventSynchronize",
                     cudaEventSynchronize(after_memcpy_out_event))
          // The memcpy out of the fusion buffer is done after this point has
          // been reached.
          ACTIVITY_END_ALL(entries, timeline)
          RELEASE_EVENT(entries, after_memcpy_out_event);
        }

        CUDA_CHECK(entries, "cudaEventSynchronize",
                   cudaEventSynchronize(done_event))
        PUSH_RANGE("horovod: Callback Loop", 6)
        for (auto it = entries.begin(); it != entries.end(); it++) {
          timeline.End(it->tensor_name, it->output);
          it->callback(Status::OK());
        }
        POP_RANGE
        RELEASE_EVENT(entries, done_event);
        POP_RANGE
      });
      finalizer_thread.detach();
      return;
    }
#endif

    PUSH_RANGE("horovod: CPU ALLREDUCE", 4)
    if (entries.size() > 1) {
      // Access the fusion buffer.
      auto& buffer = horovod_global.tensor_fusion_buffers[std::make_tuple(
          first_entry.device, first_entry.context->framework())];
      auto buffer_data = buffer->AccessData(first_entry.context);

      // Copy memory into the fusion buffer.
      ACTIVITY_START_ALL(entries, timeline, "MEMCPY_IN_FUSION_BUFFER")
      int64_t offset = 0;
      for (auto it = entries.begin(); it != entries.end(); it++) {
#if HAVE_CUDA
        if (on_gpu) {
          CUDA_CHECK(entries, "cudaMemcpyAsync",
                     cudaMemcpyAsync(
                         (void*)(buffer_data + offset),
                         (const void*)it->tensor->data(),
                         (size_t)it->tensor->size(), cudaMemcpyDeviceToDevice,
                         horovod_global.streams[first_entry.device]))
        } else {
#endif
          std::memcpy((void*)(buffer_data + offset),
                      (const void*)it->tensor->data(),
                      (size_t)it->tensor->size());
#if HAVE_CUDA
        }
#endif
        offset += it->tensor->size();
      }
#if HAVE_CUDA
      if (on_gpu) {
        CUDA_CHECK(
            entries, "cudaStreamSynchronize",
            cudaStreamSynchronize(horovod_global.streams[first_entry.device]))
      }
#endif
      ACTIVITY_END_ALL(entries, timeline)

      ACTIVITY_START_ALL(entries, timeline, "MPI_ALLREDUCE")
      int64_t num_elements = 0;
      for (auto it = entries.begin(); it != entries.end(); it++) {
        num_elements += it->tensor->shape().num_elements();
      }
      PUSH_RANGE("horovod: CPU ALLREDUCE MPI FUSED", 5)
//#ifndef USE_HYBRID_ALLREDUCE
      if (horovod_use_hybrid_allreduce == nullptr) {
        MPI_CHECK(entries, "MPI_Allreduce",
                  MPI_Allreduce(MPI_IN_PLACE, (void*)buffer_data,
                                (int)num_elements,
                                GetMPIDataType(first_entry.tensor), 
                                GetMPISumOp(first_entry.tensor),
                                MPI_COMM_WORLD))
//#else
      } else {
       hybridAllReduce_nosplit_cpu((const void*)buffer_data, (void*)buffer_data,
                                   (size_t)num_elements, GetMPIDataType(first_entry.tensor), 
                                   GetMPISumOp(first_entry.tensor), horovod_global.local_comm_socket, 
                                   horovod_global.node_comm_socket,
                                   horovod_global.local_rank_socket, horovod_global.node_size_socket);
      }
//#endif
      ACTIVITY_END_ALL(entries, timeline)
      POP_RANGE

      // Copy memory out of the fusion buffer.
      ACTIVITY_START_ALL(entries, timeline, "MEMCPY_OUT_FUSION_BUFFER")
      offset = 0;
      for (auto it = entries.begin(); it != entries.end(); it++) {
#if HAVE_CUDA
        if (on_gpu) {
          CUDA_CHECK(entries, "cudaMemcpyAsync",
                     cudaMemcpyAsync(
                         (void*)it->output->data(),
                         (const void*)(buffer_data + offset),
                         (size_t)it->tensor->size(), cudaMemcpyDeviceToDevice,
                         horovod_global.streams[first_entry.device]))
        } else {
#endif
          std::memcpy((void*)it->output->data(),
                      (const void*)(buffer_data + offset),
                      (size_t)it->tensor->size());
#if HAVE_CUDA
        }
#endif
        offset += it->tensor->size();
      }
#if HAVE_CUDA
      if (on_gpu) {
        CUDA_CHECK(
            entries, "cudaStreamSynchronize",
            cudaStreamSynchronize(horovod_global.streams[first_entry.device]))
      }
#endif
      ACTIVITY_END_ALL(entries, timeline)
    } else {
      PUSH_RANGE("horovod: CPU ALLREDUCE MPI SINGLE", 5)
      auto e = first_entry;
      ACTIVITY_START_ALL(entries, timeline, "MPI_ALLREDUCE")
//#ifndef USE_HYBRID_ALLREDUCE
      if (horovod_use_hybrid_allreduce == nullptr) {
        const void* sendbuf = e.tensor->data() == e.output->data()
                                  ? MPI_IN_PLACE
                                  : (const void*)e.tensor->data();
        MPI_CHECK(entries, "MPI_Allreduce",
                  MPI_Allreduce(sendbuf, (void*)e.output->data(),
                                (int)e.tensor->shape().num_elements(),
                                GetMPIDataType(e.tensor),
                                GetMPISumOp(e.tensor),
                                MPI_COMM_WORLD))
//#else
     } else {
       hybridAllReduce_nosplit_cpu((const void*)e.tensor->data(), (void*)e.output->data(),
                                   (size_t)e.tensor->shape().num_elements(), GetMPIDataType(e.tensor), 
                                   GetMPISumOp(e.tensor), horovod_global.local_comm_socket, 
                                   horovod_global.node_comm_socket,
                                   horovod_global.local_rank_socket, horovod_global.node_size_socket);
     }
//#endif
      ACTIVITY_END_ALL(entries, timeline)
      POP_RANGE
    }

    PUSH_RANGE("horovod: CPU Callback Loop", 6)
    for (auto it = entries.begin(); it != entries.end(); it++) {
      timeline.End(it->tensor_name, it->output);
      it->callback(Status::OK());
    }
    POP_RANGE

    POP_RANGE

  } else if (response.response_type() == MPIResponse::BROADCAST) {
    assert(entries.size() == 1);
    auto e = entries[0];

    // On root rank, MPI_Bcast sends data, on other ranks it receives data.
    void* data_ptr;
    if (horovod_global.rank == e.root_rank) {
      data_ptr = (void*)e.tensor->data();
    } else {
      data_ptr = (void*)e.output->data();
    }

    ACTIVITY_START_ALL(entries, timeline, "MPI_BCAST")
    MPI_CHECK(entries, "MPI_Bcast",
              MPI_Bcast(data_ptr, (int)e.tensor->shape().num_elements(),
                        GetMPIDataType(e.tensor), e.root_rank, MPI_COMM_WORLD))
    ACTIVITY_END_ALL(entries, timeline)

    timeline.End(e.tensor_name, e.output);
    e.callback(Status::OK());
  } else if (response.response_type() == MPIResponse::ERROR) {
    assert(entries.size() == 1);
    auto e = entries[0];

    status = Status::PreconditionError(response.error_message());
    timeline.End(e.tensor_name, nullptr);
    e.callback(status);
  }
}

// Report Tensors that were submitted to be reduced, gathered or broadcasted by
// some ranks but not others and are waiting for long time to get processed.
void CheckForStalledTensors(HorovodGlobalState& state) {
  bool preamble = false;
  std::vector<bool> ready(state.size, false);
  auto now = std::chrono::steady_clock::now();
  for (auto it = state.message_table->begin(); it != state.message_table->end();
       it++) {
    auto tensor_name = it->first;
    std::vector<MPIRequest>& messages = std::get<0>(it->second);
    std::chrono::steady_clock::time_point start_at = std::get<1>(it->second);

    if (now - start_at > STALL_WARNING_TIME) {
      if (!preamble) {
        std::cerr << "WARNING: One or more tensors were submitted to be "
                     "reduced, gathered or broadcasted by subset of ranks and "
                     "are waiting for remainder of ranks for more than "
                  << std::chrono::duration_cast<std::chrono::seconds>(
                         STALL_WARNING_TIME)
                         .count()
                  << " seconds. ";
        std::cerr << "This may indicate that different ranks are trying to "
                     "submit different tensors or that only subset of ranks is "
                     "submitting tensors, which will cause deadlock. ";
        std::cerr << "Stalled ops: ";
        preamble = true;
      } else {
        std::cerr << ", ";
      }
      std::cerr << tensor_name;
      //std::cerr << " [ready ranks:";
      //for (auto msg_iter = messages.begin(); msg_iter != messages.end();
      //     msg_iter++) {
      //  if (msg_iter == messages.begin()) {
      //    std::cerr << " ";
      //  } else {
      //    std::cerr << ", ";
      //  }
      //  std::cerr << msg_iter->request_rank();
      //}
      //std::cerr << "]";
      //std::cerr << " [stalled host:local rank:global rank:";
      std::cerr << " [stalled global rank:";
      for (int i = 0; i < state.size; i++){
        if (i == 0) {
          std::cerr << " ";
        }

        if (not ready[i]) {
          //std::cerr << state.hostlist[i].data() << ":" ;
          //std::cerr << state.local_rank_list[i] << ":" ;
          std::cerr << i << ", " ;
        }
      }
      std::cerr << "]";
    }
  }
  if (preamble) {
    std::cerr << std::endl;
  }
}

static void ConstructControlTree(int my_rank, int total_ranks,
				 int& parent_rank,
				 std::vector<int>& child_ranks)
{
  int radix = 2;
  auto horovod_control_radix = std::getenv("HOROVOD_CONTROL_RADIX");
  if (horovod_control_radix != nullptr) {
    radix = std::atol(horovod_control_radix);
  }
  if (my_rank == 0) {
    std::cout << "Using hierarchical control plane, radix = " << radix << std::endl;
    auto horovod_use_hybrid_allreduce = std::getenv("HOROVOD_USE_HYBRID_ALLREDUCE");
//#ifdef USE_HYBRID_ALLREDUCE
    if (horovod_use_hybrid_allreduce != nullptr) {
      std::cout << "Using HYBRID allreduce..." << std::endl;
//#else
    } else {
      std::cout << "Using standard allreduce..." << std::endl;
    }
//#endif
  }
  // a value of 0 requests a completely flat tree
  if(radix == 0)
    radix = total_ranks - 1;

  int prev_node = -1;
  int cur_node = 0;
  int tree_size = total_ranks;

  bool is_parent = false;
  do {
    // we should only traverse the subtrees we are in
    assert((my_rank >= cur_node) && (my_rank < (cur_node + tree_size)));

    // first node in subtree is root of subtree
    is_parent = (cur_node == my_rank);
    if(is_parent)
      parent_rank = prev_node;
    prev_node = cur_node;
    cur_node++;
    tree_size--;

    // divide remainder by tree radix, keeping track of remainder
    int base_subtree_size = tree_size / radix;
    int leftovers = tree_size % radix;

    for(int i = 0; i < radix; i++) {
      tree_size = base_subtree_size + ((i < leftovers) ? 1 : 0);
      if(tree_size == 0) break; // all later subtrees will be empty too

      if(is_parent) {
	child_ranks.push_back(cur_node);
      } else {
	assert(my_rank >= cur_node);
	if(my_rank < (cur_node + tree_size))
	  break; // we'll loop back around for the next level of the tree
      }

      cur_node += tree_size;
    }
  } while(!is_parent);

  if(std::getenv("HOROVOD_SHOW_CONTROL") != nullptr) {
    std::cout << "rank " << my_rank << ": parent=" << parent_rank
	      << " children=[";
    for (auto child : child_ranks)
      std::cout << ' ' << child;
    std::cout << " ]\n";
  }
}

// The MPI background thread loop coordinates all the MPI processes and the
// tensor reductions. The design of the communicator mechanism is limited by a
// few considerations:
//
//      1. Some MPI implementations require all MPI calls to happen from a
//      single thread. Since TensorFlow may use several threads for graph
//      processing, this means we must have our own dedicated thread for dealing
//      with MPI.
//      2. We want to gracefully handle errors, when MPI processes do not
//      properly agree upon what should happen (such as mismatched types or
//      shapes). To do so requires the MPI processes to know about the shapes
//      and types of the relevant tensors on the other processes.
//      3. The MPI reductions and gathers should be able to happen in parallel
//      with other ongoing operations. This means that they cannot be blocking
//      ops, but rather must be async ops, the execution of which happens on a
//      separate thread.
//      4. We cannot guarantee that all the MPI processes reduce their tensors
//      in the same order, so we cannot dispatch one thread per tensor,
//      otherwise we may end up dispatching many blocked threads and never make
//      progress if we have a thread pool limit.
//
// The coordinator currently follows a master-worker paradigm. Rank zero acts
// as the master (the "coordinator"), whereas all other ranks are simply
// workers. Each rank runs its own background thread which progresses in ticks.
// In each tick, the following actions happen:
//
//      a) The workers send an MPIRequest to the coordinator, indicating what
//      they would like to do (which tensor they would like to gather and
//      reduce, as well as their shape and type). They repeat this for every
//      tensor that they would like to operate on.
//
//      b) The workers send an empty "DONE" message to the coordinator to
//      indicate that there are no more tensors they wish to operate on.
//
//      c) The coordinator receives the MPIRequests from the workers, as well
//      as from its own TensorFlow ops, and stores them in a request table. The
//      coordinator continues to receive MPIRequest messages until it has
//      received MPI_SIZE number of empty "DONE" messages.
//
//      d) The coordinator finds all tensors that are ready to be reduced,
//      gathered, or all operations that result in an error. For each of those,
//      it sends an MPIResponse to all the workers. When no more MPIResponses
//      are available, it sends a "DONE" response to the workers. If the process
//      is being shutdown, it instead sends a "SHUTDOWN" response.
//
//      e) The workers listen for MPIResponse messages, processing each one by
//      doing the required reduce or gather, until they receive a "DONE"
//      response from the coordinator. At that point, the tick ends.
//      If instead of "DONE" they receive "SHUTDOWN", they exit their background
//      loop.
void BackgroundThreadLoop(HorovodGlobalState& state) {
  // Initialize MPI. This must happen on the background thread, since not all
  // MPI implementations support being called from multiple threads.
  //
  // We will ask for multiple threads, so other libraries like mpi4py can
  // be used together with Horovod if multi-threaded MPI is installed.
  int provided;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);

  // Get MPI rank to determine if we are rank zero.
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  bool is_coordinator = rank == 0;

  // Get MPI size to determine how many tensors to wait for before reducing.
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Determine local rank by querying the local communicator.
  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &local_comm);
  int local_rank, local_size;
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_size(local_comm, &local_size);

  MPI_Op_create(fp16_sum_reduce, 1, &state.Float16SumOp);

  auto horovod_use_hybrid_allreduce = std::getenv("HOROVOD_USE_HYBRID_ALLREDUCE");
//#ifdef USE_HYBRID_ALLREDUCE
  if (horovod_use_hybrid_allreduce != nullptr) {
    // Create intranode MPI communicator (connecting local ranks)
    MPI_Comm node_comm;
    MPI_Comm_split(MPI_COMM_WORLD, local_rank, rank, &node_comm);
    int node_rank, node_size;
    MPI_Comm_size(node_comm, &node_size);
    MPI_Comm_rank(node_comm, &node_rank);

    /* Setup intrasocket communicators */
    MPI_Comm local_comm_socket;
    //MPI_Comm_split(local_comm, (int)(local_rank < local_size/NSOCKETS), local_rank, &local_comm_socket);
    MPI_Comm_split(local_comm, (int)(local_rank < 4), local_rank, &local_comm_socket); // hardcode for 8 local ranks across 2 sockets
    int local_rank_socket, local_size_socket;
    MPI_Comm_size(local_comm_socket, &local_size_socket);
    MPI_Comm_rank(local_comm_socket, &local_rank_socket);

    MPI_Comm node_comm_socket;
    MPI_Comm_split(MPI_COMM_WORLD, local_rank_socket, rank, &node_comm_socket);
    int node_rank_socket, node_size_socket;
    MPI_Comm_size(node_comm_socket, &node_size_socket);
    MPI_Comm_rank(node_comm_socket, &node_rank_socket);

    state.node_size = node_size;
    state.node_rank = node_rank;
    state.local_comm = local_comm;
    state.node_comm = node_comm;

    state.local_comm_socket = local_comm_socket;
    state.local_size_socket = local_size_socket;
    state.local_rank_socket = local_rank_socket;
    state.node_comm_socket = node_comm_socket;
    state.node_size_socket = node_size_socket;
    state.node_rank_socket = node_rank_socket;

    /* Easy warning messages to make sure hybrid code is not used in an unsupported configuration. Need to generalize.*/
    /* Check for whole node multiple */
    if (size > 8 and size % 8 != 0) {
      std::cerr << "ALERT!!: A value other than 8 ranks per node detected! HOROVOD_HYBRID_ALLREDUCE not supported in this configuration. Recompile without this feature.";
      std::cerr << std::endl;
      MPI_Finalize(); exit(EXIT_FAILURE);                             \
    }

    /* Check for whole socket multiple */
    if (size > 4 and size % 4 != 0){
      std::cerr << "ALERT!!: A value other than 4 ranks per socket detected! HOROVOD_HYBRID_ALLREDUCE not supported in this configuration. Recompile without this feature.";
      std::cerr << std::endl;
      MPI_Finalize(); exit(EXIT_FAILURE);                             \
    }
  }
//#endif

  int len;
  MPI_Get_processor_name(state.hostname.data(), &len);

  if (rank == 0) {
    state.hostlist.resize(size);
    state.local_rank_list.resize(size);
  }

  MPI_Gather(state.hostname.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, state.hostlist.data(),
             MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Gather(&local_rank, 1, MPI_INT, state.local_rank_list.data(),
             1, MPI_INT, 0, MPI_COMM_WORLD);

  state.rank = rank;
  state.local_rank = local_rank;
  state.size = size;
  state.local_size = local_size;
  state.mpi_threads_supported = (provided == MPI_THREAD_MULTIPLE);
  state.initialization_done = true;

  // Open the timeline file on coordinator.
  auto horovod_timeline = std::getenv("HOROVOD_TIMELINE");
  if (is_coordinator && horovod_timeline != nullptr) {
    state.timeline.Initialize(std::string(horovod_timeline));
  }

  // Override Tensor Fusion threshold, if it's set.
  auto horovod_fusion_threshold = std::getenv("HOROVOD_FUSION_THRESHOLD");
  if (horovod_fusion_threshold != nullptr) {
    state.tensor_fusion_threshold = std::atol(horovod_fusion_threshold);
  }

  if (rank == 0) {
    std::cout << "Using HOROVOD_FUSION_THRESHOLD = " << state.tensor_fusion_threshold << std::endl;
  }

  // each rank finds its place in the tree, remembering the rank of the
  //  parent and any children
  int parent_rank;
  std::vector<int> child_ranks;

  ConstructControlTree(rank, size, parent_rank, child_ranks);

  // Initialize the tensor count table. No tensors are available yet.
  state.message_table = std::unique_ptr<MessageTable>(new MessageTable());

  // allocate buffers for the Irecv's we post for parent and child nodes
  const size_t MAX_UPSTREAM_MESSAGE_SIZE = 10*65536;
  const size_t MAX_DOWNSTREAM_MESSAGE_SIZE = 10*65536;

  char *parent_send_buffer = new char[MAX_UPSTREAM_MESSAGE_SIZE];
  char *parent_recv_buffer = new char[MAX_DOWNSTREAM_MESSAGE_SIZE];
  std::vector<char *> child_recv_buffers;
  for (size_t i = 0; i < child_ranks.size(); i++)
    child_recv_buffers.push_back(new char[MAX_UPSTREAM_MESSAGE_SIZE]);

  // we'll also need an array of MPI_Request's (the MPI thing, not
  //   horovod::common::MPIRequest) for sends and receives
  ::MPI_Request mpi_send_req = MPI_REQUEST_NULL;
  bool parent_send_in_flight = false;
  std::vector<::MPI_Request> mpi_recv_reqs(1 + child_ranks.size(),
					   MPI_REQUEST_NULL);

  // post the initial receives
  if(parent_rank != -1)
    MPI_Irecv(parent_recv_buffer, MAX_DOWNSTREAM_MESSAGE_SIZE, MPI_BYTE,
	      parent_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &mpi_recv_reqs[0]);
  for(size_t i = 0; i < child_ranks.size(); i++)
    MPI_Irecv(child_recv_buffers[i], MAX_UPSTREAM_MESSAGE_SIZE, MPI_BYTE,
	      child_ranks[i], MPI_ANY_TAG, MPI_COMM_WORLD,
	      &mpi_recv_reqs[i+1]);

  int sleep_in_ms = 5;
  {
    auto horovod_sleep_interval = std::getenv("HOROVOD_SLEEP_INTERVAL");
    if (horovod_sleep_interval != nullptr) {
      sleep_in_ms = std::atol(horovod_sleep_interval);
    }
  }

  // The coordinator sends a SHUTDOWN message to trigger shutdown.
  bool should_shut_down = false;
  // MPIRequests of tensors our subtree is ready to reduce
  std::vector<MPIRequest> ready_to_reduce;
  do {
    // This delay determines thread frequency and MPI message latency
    if (sleep_in_ms > 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_in_ms));

    // Copy the data structures from global state under this lock.
    // However, don't keep the lock for the rest of the loop, so that
    // enqueued stream callbacks can continue.
    {
      std::deque<MPIRequest> message_queue;
      {
	std::lock_guard<std::mutex> guard(state.mutex);
	message_queue.swap(state.message_queue);
      }

      // if we have children in the tree, we put messages in the table until
      //  we have received equivalent messages from every child - do this
      //  even if we have no children as the response path needs it
      for (MPIRequest &message : message_queue) {
	bool reduce = IncrementTensorCount(state.message_table, message,
					   1 + child_ranks.size());
	if (reduce) {
	  ready_to_reduce.push_back(message);
	}
      }
    }

    // next we check the status of non-blocking MPI calls
    if(size > 1) { // no need if there's only one node
      if(parent_send_in_flight) {
	int flag;
	MPI_Test(&mpi_send_req, &flag, MPI_STATUS_IGNORE);
	if(flag) parent_send_in_flight = false;
      }

      while(true) {
	int index;
	int flag = 0;
	MPI_Status status;
	MPI_Testany(mpi_recv_reqs.size(), &mpi_recv_reqs[0],
		    &index, &flag, &status);
	if(!flag) break;

        int msg_length;
        MPI_Get_count(&status, MPI_BYTE, &msg_length);
	//std::cout << "rank " << rank << " received message from " << status.MPI_SOURCE << " (" << msg_length << " bytes)\n";

	if(index == 0) {
	  // message from parent (i.e. an MPIResponse)

	  // before we parse the message ourselves, forward it on to any
	  //  children

	  // we require local completion so that we can reuse the recv
	  //  buffer
	  std::vector<MPI_Request> reqs(child_ranks.size(),
					MPI_REQUEST_NULL);
	  for (size_t i = 0; i < child_ranks.size(); i++)
	    MPI_Isend(parent_recv_buffer, msg_length, MPI_BYTE,
		      child_ranks[i], 0, MPI_COMM_WORLD, &reqs[i]);
	  MPI_Waitall(child_ranks.size(), &reqs[0], MPI_STATUSES_IGNORE);

	  std::string received_message(parent_recv_buffer, msg_length);

	  MPIResponse response;
	  MPIResponse::ParseFromString(response, received_message);

	  // now that the response is parsed, we can repost the Irecv
	  MPI_Irecv(parent_recv_buffer,
		    MAX_DOWNSTREAM_MESSAGE_SIZE, MPI_BYTE,
		    parent_rank, MPI_ANY_TAG, MPI_COMM_WORLD,
		    &mpi_recv_reqs[0]);

	  if (response.response_type() == MPIResponse::SHUTDOWN) {
	    // Background thread should shut down
	    should_shut_down = true;
	  } else {
	    // since we never formed an MPIResponse ourselves, we need to
	    //  manually remove the tensor from the message table
	    for (auto name : response.tensor_names()) {
	      auto it = state.message_table->find(name);
	      assert(it != state.message_table->end());
	      state.message_table->erase(it);
	    }

	    // Process the current message
	    PerformOperation(state.tensor_table, response);
	  }
	} else {
	  // message from child (i.e. an MPIRequestList)
	  MPIRequestList received_message_list;
	  std::string received_data(child_recv_buffers[index - 1],
				    msg_length);
	  MPIRequestList::ParseFromString(received_message_list, received_data);

	  // re-post our Irecv now that parsing is complete
	  MPI_Irecv(child_recv_buffers[index - 1],
		    MAX_UPSTREAM_MESSAGE_SIZE, MPI_BYTE,
		    child_ranks[index - 1], MPI_ANY_TAG, MPI_COMM_WORLD,
		    &mpi_recv_reqs[index]);

	  for (auto& received_message : received_message_list.requests()) {
	    bool reduce = IncrementTensorCount(state.message_table,
					       received_message,
					       1 + child_ranks.size());
	    if (reduce) {
	      ready_to_reduce.push_back(received_message);
	    }
	  }
	}
      }
    }

    // with all incoming messages handled, consider sending new messages
    //  (or responses) out
    if(!ready_to_reduce.empty()) {
      if(parent_rank == -1) {
	// initiate reduction operations for anything in ready_to_reduce

	// At this point, rank zero should have a fully updated tensor count
	// table and should know all the tensors that need to be reduced or
	// gathered, and everyone else should have sent all their information
	// to rank zero. We can now do reductions and gathers; rank zero will
	// choose which ones and in what order, and will notify the other ranks
	// before doing each reduction.
	std::vector<MPIResponse> responses;
	for (auto it = ready_to_reduce.begin(); it != ready_to_reduce.end();
	     it++) {
	  MPIResponse response = ConstructMPIResponse(state.message_table,
						      it->tensor_name());
	  responses.push_back(std::move(response));
	}
	// consumed all the entries in the list
	ready_to_reduce.clear();

	while (!responses.empty()) {
	  auto it = responses.begin();
	  MPIResponse response = *it;
	  assert(response.tensor_names().size() == 1);
	  it = responses.erase(it);

	  if (response.response_type() == MPIResponse::ResponseType::ALLREDUCE) {
	    // Attempt to add more responses to this fused response.
	    auto& entry = state.tensor_table[response.tensor_names()[0]];
	    int64_t tensor_size = entry.tensor->size();

	    while (it != responses.end()) {
	      assert(it->tensor_names().size() == 1);
	      auto& new_entry = state.tensor_table[it->tensor_names()[0]];
	      int64_t new_tensor_size = new_entry.tensor->size();

	      if (response.response_type() == it->response_type() &&
		  response.devices() == it->devices() &&
		  entry.tensor->dtype() == new_entry.tensor->dtype() &&
		  ((tensor_size + new_tensor_size) <=
		   state.tensor_fusion_threshold)) {
		// These tensors will fuse together well.
		tensor_size += new_tensor_size;
		response.add_tensor_names(it->tensor_names()[0]);
		it = responses.erase(it);
	      } else {
		// Don't try to fuse additional tensors since they are usually
		// computed in order of requests and skipping tensors may mean
		// that the batch will have to wait longer while skipped tensors
		// could be reduced at that time.
		break;
	      }
	    }
	  }

	  // send the list of tensors to reduce just to our direct child
	  //  ranks - they'll forward it on

	  std::string encoded_response;
	  MPIResponse::SerializeToString(response, encoded_response);

	  // we require local completion so that we can reuse the recv
	  //  buffer
	  std::vector<MPI_Request> reqs(child_ranks.size(),
					MPI_REQUEST_NULL);
	  for (size_t i = 0; i < child_ranks.size(); i++)
	    MPI_Isend(encoded_response.c_str(), encoded_response.length(),
		      MPI_BYTE,
		      child_ranks[i], 0, MPI_COMM_WORLD, &reqs[i]);
	  MPI_Waitall(child_ranks.size(), &reqs[0], MPI_STATUSES_IGNORE);

	  // Perform the collective operation. All nodes should end up performing
	  // the same operation.
	  PerformOperation(state.tensor_table, response);
	}
      } else {
	// send a message to our parent unless we already have one
	//  in flight
	if(!parent_send_in_flight) {
	  MPIRequestList message_list;
	  for (MPIRequest& message : ready_to_reduce)
	    message_list.add_requests(message);
	  // consumed all the entries in the list
	  ready_to_reduce.clear();

	  // TODO: get rid of std::string intermediate
	  std::string encoded_message;
	  MPIRequestList::SerializeToString(message_list, encoded_message);
	  size_t msg_length = encoded_message.size();
	  if(msg_length > MAX_UPSTREAM_MESSAGE_SIZE) {
	    std::cerr << "rank " << rank << ": upstream message too large (" << msg_length << " > " << MAX_UPSTREAM_MESSAGE_SIZE << ")\n";
	    assert(0);
	  }
	  memcpy(parent_send_buffer, encoded_message.c_str(),
		 msg_length);
	  // use Issend here so that completion does not occur until the
	  //  message is delivered to parent - limits us to 1 in flight
	  MPI_Issend(parent_send_buffer, msg_length, MPI_BYTE,
		     parent_rank, 0, MPI_COMM_WORLD, &mpi_send_req);
	  parent_send_in_flight = true;
	}
      }
    }

    // if it's time to shut down, let children know
    if(state.shut_down) {
      should_shut_down = true;

      // Notify all nodes that we are done with the reductions for this tick.
      MPIResponse done_response;
      done_response.set_response_type(MPIResponse::SHUTDOWN);

      std::string encoded_response;
      MPIResponse::SerializeToString(done_response, encoded_response);

      // blocking sends are fine here...
      for (size_t i = 0; i < child_ranks.size(); i++)
	MPI_Send(encoded_response.c_str(), encoded_response.length(),
		 MPI_BYTE,
		 child_ranks[i], 0, MPI_COMM_WORLD);
    }

    // Check for stalled tensors.
    if (std::chrono::steady_clock::now() - state.last_stall_check >
	STALL_WARNING_TIME) {
      //CheckForStalledTensors(state);
      state.last_stall_check = std::chrono::steady_clock::now();
    }
  } while (!should_shut_down);

  // TODO: init.cu:645 WARN Cuda failure 'driver shutting down'
  //#if HAVE_NCCL
  //  for (auto it = horovod_global.streams.begin();
  //       it != horovod_global.streams.end(); it++) {
  //    cudaStreamSynchronize(it->second);
  //  }
  //  for (auto it = horovod_global.nccl_comms.begin();
  //       it != horovod_global.nccl_comms.end(); it++) {
  //    ncclCommDestroy(it->second);
  //  }
  //#endif
  MPI_Op_free(&state.Float16SumOp);
  MPI_Finalize();
}

// Start Horovod background thread. Ensure that this is
// only done once no matter how many times this function is called.
void InitializeHorovodOnce() {
  // Ensure background thread is only started once.
  if (!horovod_global.initialize_flag.test_and_set()) {
    horovod_global.background_thread =
        std::thread(BackgroundThreadLoop, std::ref(horovod_global));
  }

  // Wait to ensure that the background thread has finished initializing MPI.
  while (!horovod_global.initialization_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

} // namespace

Status CheckInitialized() {
  if (!horovod_global.initialization_done) {
    return Status::PreconditionError(
        "Horovod has not been initialized; use hvd.init().");
  }
  return Status::OK();
}

extern "C" {

void horovod_init() { InitializeHorovodOnce(); }

int horovod_rank() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.rank;
}

int horovod_local_rank() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.local_rank;
}

int horovod_size() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.size;
}

int horovod_local_size() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.local_size;
}

int horovod_mpi_threads_supported() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.mpi_threads_supported ? 1 : 0;
}
}

// MPI must be initialized and the background thread must be running before
// this function is called.
void EnqueueTensorAllreduce(std::shared_ptr<OpContext> context,
                            std::shared_ptr<Tensor> tensor,
                            std::shared_ptr<Tensor> output,
                            std::shared_ptr<ReadyEvent> ready_event,
                            const std::string name, const int device,
                            StatusCallback callback) {
  PUSH_RANGE("EnqueueTensor", 13)
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPIRequest message;
  message.set_request_rank(rank);
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_root_rank(0);
  message.set_device(device);
  message.set_request_type(MPIRequest::ALLREDUCE);
  for (int i = 0; i < tensor->shape().dims(); i++) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.output = output;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  horovod_global.tensor_table.emplace(name, std::move(e));
  horovod_global.message_queue.push_back(message);
  POP_RANGE
}

// MPI must be initialized and the background thread must be running before
// this function is called.
void EnqueueTensorAllgather(std::shared_ptr<OpContext> context,
                            std::shared_ptr<Tensor> tensor,
                            std::shared_ptr<ReadyEvent> ready_event,
                            const std::string name, const int device,
                            StatusCallback callback) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPIRequest message;
  message.set_request_rank(rank);
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_root_rank(0);
  message.set_device(device);
  message.set_request_type(MPIRequest::ALLGATHER);
  for (int i = 0; i < tensor->shape().dims(); i++) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  horovod_global.tensor_table.emplace(name, std::move(e));
  horovod_global.message_queue.push_back(message);
}

// MPI must be initialized and the background thread must be running before
// this function is called.
void EnqueueTensorBroadcast(std::shared_ptr<OpContext> context,
                            std::shared_ptr<Tensor> tensor,
                            std::shared_ptr<Tensor> output, int root_rank,
                            std::shared_ptr<ReadyEvent> ready_event,
                            const std::string name, const int device,
                            StatusCallback callback) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPIRequest message;
  message.set_request_rank(rank);
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_root_rank(root_rank);
  message.set_device(device);
  message.set_request_type(MPIRequest::BROADCAST);
  for (int i = 0; i < tensor->shape().dims(); i++) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.output = output;
  e.root_rank = root_rank;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  horovod_global.tensor_table.emplace(name, std::move(e));
  horovod_global.message_queue.push_back(message);
}

} // namespace common
} // namespace horovod
