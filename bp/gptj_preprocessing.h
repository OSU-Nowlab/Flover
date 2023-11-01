
#pragma once

#include <vector>

#include "src/flover/kernels/add_residual_kernels.h"
#include "src/flover/kernels/layernorm_kernels.h"
#include "src/flover/layers/BaseLayer.h"
#include "src/flover/layers/FfnLayer.h"
#include "src/flover/layers/attention_layers/BaseAttentionLayer.h"
#include "src/flover/models/gptj/GptJDecoderLayerWeight.h"
#include "src/flover/utils/Tensor.h"
#include "src/flover/utils/allocator.h"
#include "src/flover/utils/cublasMMWrapper.h"
#include "src/flover/utils/custom_ar_comm.h"
#include "src/flover/utils/nccl_utils.h"

#include "src/flover/kernels/bert_preprocess_kernels.h"
#include "src/flover/layers/TensorParallelGeluFfnLayer.h"
#include "src/flover/layers/attention_layers/TensorParallelGptContextAttentionLayer.h"
#include "src/flover/kernels/decoding_kernels.h"

namespace flover {
    

template<typename T>
class GptJContextDecoder: public BaseLayer {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_    = 0;

    // meta data
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;
    size_t rotary_embedding_dim_;
    bool   neox_rotary_style_;  // A unify way for GPT-NeoX in the future, not used now.

    float  layernorm_eps_;
    size_t local_head_num_;
    size_t vocab_size_;
    size_t vocab_size_padded_;

    // calculated data
    size_t hidden_units_;

    NcclParam tensor_para_;
    NcclParam pipeline_para_;

    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    int                                 enable_custom_all_reduce_;

    AttentionType attention_type_;

    bool is_qk_buf_float_;


    bool isValidLayerParallelId(uint l);
    bool isFirstLayerParallelId(uint l);
    bool isLastLayerParallelId(uint l);
    int  getFirstLayerParallelId();

    void initialize();
    

protected:
       

public:
    BaseAttentionLayer<T>* self_attention_layer_;
    FfnLayer<T>*           ffn_layer_;

    T*        decoder_normed_input_   = nullptr;
    T*        self_attn_output_       = nullptr;
    T*        ffn_output_             = nullptr;
    T*        decoder_layer_output_   = nullptr;
    size_t*   h_pinned_token_num_ptr_ = nullptr;
    int*      padding_offset_         = nullptr;
    int*      cu_seqlens_             = nullptr;
    uint32_t* seq_limit_len_          = nullptr;

    /*shared with GptJ*/
    bool                                  has_prefix_prompt_;
    bool                                  has_prefix_soft_prompt_;
    PromptLearningType                    prompt_learning_type_;
    int*                                  output_ids_buf_;
    int*                                  parent_ids_buf_;
    int*                                  tiled_total_padding_count_ = nullptr;
    int*                                  cache_indirections_[2]     = {nullptr, nullptr};
    const T**                             prompt_learning_weight_batch_;
    int*                                  tiled_prompt_lengths_buf_;  // only needed by prefix prompts
    int*                                  tiled_input_ids_buf_;
    int*                                  tiled_input_lengths_buf_;
    T*                                    context_decoder_input_buf_;
    T*                                    context_decoder_output_buf_;
    T*                                    input_attention_mask_;
    T*                                    key_cache_;
    T*                                    value_cache_;
    T*                                    decoder_output_buf_;
    bool*                                 finished_buf_;
    float*                                cum_log_probs_;
    int*                                  start_ids_buf_;
    T*                                    padded_embedding_kernel_;
    T*                                    padded_embedding_bias_;
    const T*                              padded_embedding_kernel_ptr_;
    const T*                              padded_embedding_bias_ptr_;
    T*                                    decoder_input_buf_;
    
    T*                                    normed_decoder_output_buf_;
    float*                                logits_buf_;
    float*                                nccl_logits_buf_;
    bool*                                 h_finished_buf_;
    int*                                  end_ids_buf_;
    float*                                output_log_probs_buf_;
    int*                                  transposed_output_ids_buf_;

    bool*                                 generation_should_stop_ = nullptr;

    int*                                  sequence_lengths_          = nullptr;
    bool*                                 masked_tokens_             = nullptr;

    GptJContextDecoder(size_t                                                 max_batch_size,
                       size_t                                                 max_seq_len,
                       size_t                                                 head_num,
                       size_t                                                 size_per_head,
                       size_t                                                 inter_size,
                       size_t                                                 num_layer,
                       size_t                                                 rotary_embedding_dim,
                       bool                                                   neox_rotary_style,
                       float                                                  layernorm_eps,
                       NcclParam                                              tensor_para,
                       NcclParam                                              pipeline_para,
                       cudaStream_t                                           stream,
                       cublasMMWrapper*                                       cublas_wrapper,
                       bool                                                   is_free_buffer_after_forward,
                       bool                                                   is_qk_buf_float,
                       AttentionType                                          attention_type            = AttentionType::UNFUSED_MHA,
                       std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm    = nullptr,
                       int                                                    enable_custom_all_reduce_ = 0);

    GptJContextDecoder(GptJContextDecoder<T> const& decoder);

    ~GptJContextDecoder();

    void run(std::unordered_map<std::string, Tensor>*       output_tensors,
             const std::unordered_map<std::string, Tensor>* input_tensors, 
             const GptJWeight<T>*        gpt_weights);
    void forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors,
                 const std::vector<GptJDecoderLayerWeight<T>>*  gpt_decoder_layer_weight);
 
};

}