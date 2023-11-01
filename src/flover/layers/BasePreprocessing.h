#pragma once

#include <assert.h>

#include "src/flover/layers/BaseLayer.h"
#include "src/flover/utils/Tensor.h"
#include "src/flover/utils/allocator.h"
#include "src/flover/utils/cublasMMWrapper.h"

namespace flover {

class BasePreprocessing : public BaseLayer {
public:
    BasePreprocessing(cudaStream_t     stream,
              cublasMMWrapper* cublas_wrapper,
              IAllocator*      allocator,
              bool             is_free_buffer_after_forward,
              cudaDeviceProp*  cuda_device_prop = nullptr,
              bool             sparse           = false):
        stream_(stream),
        cublas_wrapper_(cublas_wrapper),
        allocator_(allocator),
        cuda_device_prop_(cuda_device_prop),
        is_free_buffer_after_forward_(is_free_buffer_after_forward),
        sparse_(sparse){};
}
}