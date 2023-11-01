/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "src/flover/layers/FfnLayer.h"
#include "src/flover/utils/custom_ar_comm.h"
#include "src/flover/utils/nccl_utils.h"

namespace flover {

template<typename T>
class TensorParallelSiluFfnLayer: public SiluFfnLayer<T> {
private:
    NcclParam                           tensor_para_;
    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    int                                 enable_custom_all_reduce_;
    bool                                do_all_reduce_;

protected:
public:
    TensorParallelSiluFfnLayer(size_t                              max_batch_size,
                               size_t                              max_seq_len,
                               size_t                              head_num,
                               size_t                              size_per_head,
                               size_t                              expert_num,
                               size_t                              inter_size,
                               NcclParam                           tensor_para,
                               cudaStream_t                        stream,
                               cublasMMWrapper*                    cublas_wrapper,
                               IAllocator*                         allocator,
                               bool                                do_all_reduce,
                               bool                                is_free_buffer_after_forward,
                               bool                                is_sparse,
                               bool                                use_gated_activation     = false,
                               std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm   = nullptr,
                               int                                 enable_custom_all_reduce = 0,
                               int                                 int8_mode                = 0);

    TensorParallelSiluFfnLayer(TensorParallelSiluFfnLayer<T> const& ffn_layer);

    void forward(std::vector<flover::Tensor>*       output_tensors,
                 const std::vector<flover::Tensor>* input_tensors,
                 const FfnWeight<T>*                           ffn_weights) override;
    void forward(TensorMap* output_tensors, TensorMap* input_tensors, const FfnWeight<T>* ffn_weights) override;
};

}  // namespace flover
