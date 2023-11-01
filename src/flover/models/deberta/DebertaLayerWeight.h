/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/flover/kernels/layernorm_kernels.h"
#include "src/flover/layers/FfnWeight.h"
#include "src/flover/layers/attention_layers/AttentionWeight.h"
#include "src/flover/models/BaseWeight.h"
#include "src/flover/utils/cublasMMWrapper.h"
#include "src/flover/utils/memory_utils.h"
#include <unordered_map>

namespace flover {

template<typename T>
struct DebertaLayerWeight {

    DebertaLayerWeight() = default;
    DebertaLayerWeight(const size_t hidden_units,
                       const size_t inter_size,
                       const size_t tensor_para_size,
                       const size_t tensor_para_rank);
    DebertaLayerWeight(const int hidden_units, const int inter_size): DebertaLayerWeight(hidden_units, inter_size, 1, 0)
    {
    }
    ~DebertaLayerWeight();
    DebertaLayerWeight(const DebertaLayerWeight& other);
    DebertaLayerWeight& operator=(const DebertaLayerWeight& other);

#ifdef SPARSITY_ENABLED
    void compress_weights(cublasMMWrapper& cublas_wrapper, int hidden_dim);
#endif

    AttentionWeight<T> attention_weights;
    LayerNormWeight<T> attn_layernorm_weights;
    FfnWeight<T>       ffn_weights;
    LayerNormWeight<T> ffn_layernorm_weights;

    void loadModel(std::string dir_path, FtCudaDataType model_file_type);

private:
    void setWeightPtr();

    size_t                                       hidden_units_;
    size_t                                       inter_size_;
    size_t                                       tensor_para_size_;
    size_t                                       tensor_para_rank_;
    bool                                         is_maintain_buffer_ = false;
    std::unordered_map<std::string, FtWeight<T>> weights_ptr_;
    T*                                           sp_weights_ptr_[6];
    bool                                         is_maintain_sp_buffer_ = false;
};

}  // namespace flover
