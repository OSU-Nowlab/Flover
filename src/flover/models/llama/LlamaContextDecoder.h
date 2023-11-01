/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <vector>

#include "src/flover/kernels/add_residual_kernels.h"
#include "src/flover/kernels/layernorm_kernels.h"
#include "src/flover/layers/BaseLayer.h"
#include "src/flover/layers/FfnLayer.h"
#include "src/flover/layers/attention_layers/BaseAttentionLayer.h"
#include "src/flover/models/llama/LlamaDecoderLayerWeight.h"
#include "src/flover/utils/Tensor.h"
#include "src/flover/utils/allocator.h"
#include "src/flover/utils/cublasMMWrapper.h"
#include "src/flover/utils/custom_ar_comm.h"
#include "src/flover/utils/nccl_utils.h"

namespace flover {

template<typename T>
class LlamaContextDecoder: public BaseLayer {
private:
    // meta data
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;
    size_t rotary_embedding_dim_;
    bool   neox_rotary_style_;
    bool   use_gptj_residual_;
    float  layernorm_eps_;

    // calculated data
    size_t hidden_units_;

    NcclParam tensor_para_;
    NcclParam pipeline_para_;

    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    int                                 enable_custom_all_reduce_;

    AttentionType attention_type_;

    int int8_mode_ = 0;

    bool is_qk_buf_float_;

    BaseAttentionLayer<T>* self_attention_layer_;
    FfnLayer<T>*           ffn_layer_;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len, bool use_shared_contexts);
    void freeBuffer() override;

    bool isValidLayerParallelId(uint l);
    bool isFirstLayerParallelId(uint l);
    bool isLastLayerParallelId(uint l);
    int  getFirstLayerParallelId();

    void initialize();

protected:
    T*      decoder_normed_input_   = nullptr;
    T*      self_attn_output_       = nullptr;
    T*      ffn_output_             = nullptr;
    T*      decoder_layer_output_   = nullptr;
    size_t* h_pinned_token_num_ptr_ = nullptr;
    int*    padding_offset_         = nullptr;
    int*    cu_seqlens_             = nullptr;

    T*   compact_decoder_features_ = nullptr;
    T*   compact_attention_mask_   = nullptr;
    int* compact_input_lengths_    = nullptr;
    T*   k_cache_layer_            = nullptr;
    T*   v_cache_layer_            = nullptr;

public:
    LlamaContextDecoder(size_t                              head_num,
                        size_t                              size_per_head,
                        size_t                              inter_size,
                        size_t                              num_layer,
                        size_t                              rotary_embedding_dim,
                        bool                                neox_rotary_style,
                        bool                                use_gptj_residual,
                        float                               layernorm_eps,
                        NcclParam                           tensor_para,
                        NcclParam                           pipeline_para,
                        cudaStream_t                        stream,
                        cublasMMWrapper*                    cublas_wrapper,
                        IAllocator*                         allocator,
                        bool                                is_free_buffer_after_forward,
                        bool                                is_qk_buf_float,
                        AttentionType                       attention_type            = AttentionType::FUSED_MHA,
                        int                                 int8_mode                = 0,
                        std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm    = nullptr,
                        int                                 enable_custom_all_reduce_ = 0);

    LlamaContextDecoder(LlamaContextDecoder<T> const& decoder);

    ~LlamaContextDecoder();

    void forward(std::vector<Tensor>*                           output_tensors,
                 const std::vector<Tensor>*                     input_tensors,
                 const std::vector<LlamaDecoderLayerWeight<T>*>* decoder_layer_weights);

    void forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors,
                 const std::vector<LlamaDecoderLayerWeight<T>*>* gpt_decoder_layer_weight);
};

}  // namespace flover