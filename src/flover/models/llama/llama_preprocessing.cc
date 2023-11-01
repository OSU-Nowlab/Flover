#include "llama_preprocessing.h"
#include "src/flover/kernels/gpt_kernels.h"


namespace flover {
template<typename T>
void LlamaPreprocessing<T>::initialize()
{
    self_attention_layer_ = new TensorParallelGptContextAttentionLayer<T>(max_batch_size_,
                                                                          max_seq_len_,
                                                                          head_num_,
                                                                          size_per_head_,
                                                                          rotary_embedding_dim_,
                                                                          neox_rotary_style_,
                                                                          tensor_para_,
                                                                          stream_,
                                                                          cublas_wrapper_,
                                                                          nullptr,
                                                                          !use_gptj_residual_,
                                                                          is_free_buffer_after_forward_,
                                                                          is_qk_buf_float_,
                                                                          false,
                                                                          int8_mode_,
                                                                          custom_all_reduce_comm_,
                                                                          enable_custom_all_reduce_);

    ffn_layer_ = new TensorParallelSiluFfnLayer<T>(max_batch_size_,
                                                   max_seq_len_,
                                                   head_num_,
                                                   size_per_head_,
                                                   0,  // expert_num
                                                   inter_size_,
                                                   tensor_para_,
                                                   stream_,
                                                   cublas_wrapper_,
                                                   nullptr,
                                                   !use_gptj_residual_,
                                                   is_free_buffer_after_forward_,
                                                   false,
                                                   true,  // use_gated_activation = false;
                                                   custom_all_reduce_comm_,
                                                   enable_custom_all_reduce_,
                                                   int8_mode_);
}


template<typename T>
void LlamaPreprocessing<T>::allocateBuffer()
{
}


template<typename T>
void LlamaPreprocessing<T>::freeBuffer()
{
}



template<typename T>
bool LlamaPreprocessing<T>::isValidLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T>
bool LlamaPreprocessing<T>::isFirstLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T>
bool LlamaPreprocessing<T>::isLastLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T>
int LlamaPreprocessing<T>::getFirstLayerParallelId()
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T>
LlamaPreprocessing<T>::LlamaPreprocessing(size_t                                                 max_batch_size,
                                          size_t                                                 max_seq_len,
                                          size_t                                                 head_num,
                                          size_t                                                 size_per_head,
                                          size_t                                                 inter_size,
                                          size_t                                                 num_layer,
                                          size_t                                                 rotary_embedding_dim,
                                          bool                                                   neox_rotary_style,
                                          bool                                                   use_gptj_residual,
                                          float                                                  layernorm_eps,
                                          NcclParam                                              tensor_para,
                                          NcclParam                                              pipeline_para,
                                          cudaStream_t                                           stream,
                                          cublasMMWrapper*                                       cublas_wrapper,
                                          bool                                                   is_free_buffer_after_forward,
                                          bool                                                   is_qk_buf_float,
                                          AttentionType                                          attention_type,
                                          int                                                    int8_mode,
                                          std::shared_ptr<AbstractCustomComm>                    custom_all_reduce_comm,
                                          int                                                    enable_custom_all_reduce):
    BaseLayer(stream, cublas_wrapper, nullptr, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    rotary_embedding_dim_(rotary_embedding_dim),
    neox_rotary_style_(neox_rotary_style),
    use_gptj_residual_(use_gptj_residual),
    layernorm_eps_(layernorm_eps),
    hidden_units_(head_num * size_per_head),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    is_qk_buf_float_(is_qk_buf_float),
    attention_type_(attention_type),
    int8_mode_(int8_mode),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    initialize();
}

template<typename T>
LlamaPreprocessing<T>::LlamaPreprocessing(LlamaPreprocessing<T> const& decoder):
    BaseLayer(decoder.stream_, decoder.cublas_wrapper_, nullptr, decoder.is_free_buffer_after_forward_),
    max_batch_size_(decoder.max_batch_size_),
    max_seq_len_(decoder.max_seq_len_),
    head_num_(decoder.head_num_),
    size_per_head_(decoder.size_per_head_),
    inter_size_(decoder.inter_size_),
    num_layer_(decoder.num_layer_),
    rotary_embedding_dim_(decoder.rotary_embedding_dim_),
    neox_rotary_style_(decoder.neox_rotary_style_),
    use_gptj_residual_(decoder.use_gptj_residual_),
    layernorm_eps_(decoder.layernorm_eps_),
    hidden_units_(decoder.hidden_units_),
    tensor_para_(decoder.tensor_para_),
    pipeline_para_(decoder.pipeline_para_),
    is_qk_buf_float_(decoder.is_qk_buf_float_),
    attention_type_(decoder.attention_type_),
    int8_mode_(decoder.int8_mode_),
    custom_all_reduce_comm_(decoder.custom_all_reduce_comm_),
    enable_custom_all_reduce_(decoder.enable_custom_all_reduce_)
{
    initialize();
}

template<typename T>
LlamaPreprocessing<T>::~LlamaPreprocessing()
{
    delete self_attention_layer_;
    delete ffn_layer_;
}


template<typename T>
void LlamaPreprocessing<T>::run(std::unordered_map<std::string, Tensor>*       output_tensors,
                                const std::unordered_map<std::string, Tensor>* input_tensors, 
                                const LlamaWeight<T>*        gpt_weights)
{   
    FT_CHECK_WITH_INFO(input_tensors->size() >= 3, "input_tensors->size() >= 3");
    FT_CHECK_WITH_INFO(output_tensors->size() >= 2, "output_tensors->size() >= 2");
    FT_CHECK(input_tensors->at("input_ids").shape.size() == 2);
    FT_CHECK(input_tensors->at("input_lengths").shape.size() == 1);
    FT_CHECK(input_tensors->find("output_seq_len") != input_tensors->end()
             && input_tensors->at("output_seq_len").shape.size() == 1);
    FT_CHECK(output_tensors->at("output_ids").shape.size() == 3);
    FT_CHECK(output_tensors->at("sequence_length").shape.size() == 2);
    FT_CHECK_WITH_INFO(input_tensors->at("input_ids").shape[0] == output_tensors->at("output_ids").shape[0],
                       "input_tensors->at(\"input_ids\").shape[0] == output_tensors->at(\"output_ids\").shape[0]");

    local_head_num_ = head_num_ / tensor_para_.world_size_;
    const size_t batch_size = output_tensors->at("output_ids").shape[0];
    const size_t beam_width = output_tensors->at("output_ids").shape[1];

    PromptLearningType request_prompt_type = PromptLearningType::no_prompt;
    int                valid_prompt_inputs = input_tensors->count("request_prompt_type")
                              + input_tensors->count("request_prompt_lengths")
                              + input_tensors->count("request_prompt_embedding");
    
    if (valid_prompt_inputs == 3) {
        request_prompt_type = static_cast<PromptLearningType>(input_tensors->at("request_prompt_type").getVal<int>());
        FT_LOG_INFO("Apply prompt embedding from input, will ignore task name ids");
    }
    else if (valid_prompt_inputs > 0) {
        FT_LOG_WARNING(
            "Prompts not applied: request_prompt_embedding, request_prompt_lengths, request_prompt_type are all needed!");
    }
    if (request_prompt_type == PromptLearningType::prefix_prompt) {
        FT_LOG_WARNING("Request prompt doesn't support prefix prompt currently!");
    }

    // Prefix Prompt Inputs
    // Padding works as follows: p p x x i i i x x --> p p i i i x x x x (p denotes prompt, i denotes input, x denotes
    // pad)
    // TODO (perkzz): move unnecessary paddings
    const int* prompt_learning_task_name_ids =
        input_tensors->count("prompt_learning_task_name_ids") ?
            input_tensors->at("prompt_learning_task_name_ids").getPtr<const int>() :
            nullptr;
    has_prefix_prompt_ =
        (prompt_learning_task_name_ids != nullptr) && (prompt_learning_type_ == PromptLearningType::prefix_prompt);
    int max_prefix_prompt_length = 0;

    FT_CHECK_WITH_INFO(
        !(prompt_learning_task_name_ids != nullptr
          && (prompt_learning_type_ == PromptLearningType::no_prompt
              || prompt_learning_type_ == PromptLearningType::soft_prompt)),
        "prompt_learning_type is prefix_prompt either p_prompt_tuning when prompt_learning_task_name_ids are provided.");

    // NOTE: Prefix Prompt PreProcessing
    // get prefix_prompt_weight for each batch --> shape [batch, beam_width]
    // --> ptrs with shape [num_layers, 2, num_heads, perfix_seq_len, size_per_head]
    std::vector<const T*> prefix_prompt_weight_batch_ptrs;
    std::vector<int>      prefix_prompt_lengths;
    if (has_prefix_prompt_) {
        for (int bs_id = 0; bs_id < batch_size; ++bs_id) {
            int task_id = prompt_learning_task_name_ids[bs_id];
            // throw errors when prompt task_name_ids are not found
            std::pair<const T*, int> prefix_prompt_weight_length_pair;
            try {
                prefix_prompt_weight_length_pair = gpt_weights->prompt_learning_table.at(task_id);
            }
            catch (const std::out_of_range& oor) {
                FT_LOG_ERROR("prefix_prompt_weights_lengths not found for prompt task id: " + task_id);
                throw oor;
            }
            for (int bw_id = 0; bw_id < beam_width; ++bw_id) {
                prefix_prompt_weight_batch_ptrs.push_back(prefix_prompt_weight_length_pair.first);
                prefix_prompt_lengths.push_back(prefix_prompt_weight_length_pair.second);
            }
        }

        max_prefix_prompt_length = *max_element(prefix_prompt_lengths.begin(), prefix_prompt_lengths.end());

        FT_LOG_DEBUG("max_prefix_prompt_length: %d", max_prefix_prompt_length);

        if (max_prefix_prompt_length == 0) {
            has_prefix_prompt_ = false;
            FT_LOG_DEBUG("prompts are not applied !");
        }
    }

    int max_input_length = input_tensors->at("input_ids").shape[1];
    FT_CHECK_WITH_INFO(!(max_input_length == 0 && max_prefix_prompt_length > 0),
                       "Prefix Prompt should come with inputs!");

    // Prefix Soft Prompt (only support request prompt embedding currently)
    has_prefix_soft_prompt_ = request_prompt_type == PromptLearningType::soft_prompt;
    const size_t max_prefix_soft_prompt_length =
        has_prefix_soft_prompt_ ? input_tensors->at("request_prompt_embedding").shape[1] : 0;
    const size_t limit_len_offset   = max_prefix_soft_prompt_length + (max_input_length == 0 ? 1 : 0);
    const size_t max_output_seq_len = input_tensors->at("output_seq_len").max<uint32_t>() + limit_len_offset;
    const size_t max_seq_len        = max_output_seq_len;
    
    // max cache seq len should include max prefix prompt length as it has k/v states
    const size_t max_cache_seq_len = max_output_seq_len + max_prefix_prompt_length;

    if (max_cache_seq_len < max_seq_len) {
        FT_LOG_WARNING("max_cache_seq_len (%d) is less than max_seq_len (%d). "
                       "Note that this reduces the memory cost of k/v cache, but may hurt the accuracy.",
                       max_cache_seq_len,
                       max_seq_len);
    }
    else if (max_cache_seq_len > max_seq_len) {
        FT_LOG_WARNING("max_cache_seq_len (%d) is larger than max_seq_len (%d). "
                       "This may lead to additional memory cost. Suggest to use smaller max_cache_seq_len.",
                       max_cache_seq_len,
                       max_seq_len);
    }

    setSeqLimitLen(seq_limit_len_, input_tensors->at("output_seq_len"), limit_len_offset, batch_size);

    const DataType       data_type      = getTensorType<T>();
    const cudaDataType_t gemm_data_type = getCudaDataType<T>();

    sync_check_cuda_error();
    // {
        TensorMap input_map(*input_tensors);
    //     // dynamic_decode_layer_->setup(batch_size, beam_width, &input_map);
        handleOptArg(&input_map, "start_id", start_ids_buf_, start_id_, batch_size);
        handleOptArg(&input_map, "end_id", end_ids_buf_, end_id_, batch_size);
    // }

    const std::vector<size_t> self_k_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                    batch_size * beam_width,
                                                    local_head_num_,
                                                    size_per_head_ / (16 / sizeof(T)),
                                                    max_cache_seq_len,
                                                    16 / sizeof(T)};
    const std::vector<size_t> self_v_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                    batch_size * beam_width,
                                                    local_head_num_,
                                                    max_cache_seq_len,
                                                    size_per_head_};

    // initialize the output ids and parent ids
    cudaMemsetAsync(output_ids_buf_, 0, sizeof(int) * batch_size * beam_width * max_seq_len, stream_);
    cudaMemsetAsync(parent_ids_buf_, 0, sizeof(int) * batch_size * beam_width * max_seq_len, stream_);
    cudaMemsetAsync(masked_tokens_, false, sizeof(bool) * batch_size * beam_width * max_cache_seq_len, stream_);
    cudaMemsetAsync(tiled_total_padding_count_, 0, sizeof(int) * batch_size * beam_width, stream_);
    
    if (beam_width > 1) {
        cudaMemsetAsync(cache_indirections_[0], 0, sizeof(int) * batch_size * beam_width * max_seq_len, stream_);
        cudaMemsetAsync(cache_indirections_[1], 0, sizeof(int) * batch_size * beam_width * max_seq_len, stream_);
    }

    int  compact_size;
    bool use_shared_contexts = (shared_contexts_ratio_ > 0.0f) && (max_input_length >= 1) && (batch_size > 1);
    if (use_shared_contexts) {
        invokeFindContextDups(shared_contexts_idx_,
                              batch_to_compact_idx_,
                              compact_idx_,
                              compact_size_,
                              input_tensors->at("input_ids").getPtr<int>(),
                              batch_size,
                              beam_width,
                              max_input_length,
                              stream_);
        cudaD2Hcpy(&compact_size, compact_size_, 1);
        use_shared_contexts = compact_size <= shared_contexts_ratio_ * batch_size;
        sync_check_cuda_error();
    }

    // Prefix prompts
    if (has_prefix_prompt_) {
        cudaMemcpyAsync(prompt_learning_weight_batch_,
                        prefix_prompt_weight_batch_ptrs.data(),
                        sizeof(T*) * batch_size * beam_width,
                        cudaMemcpyDefault,
                        stream_);
        cudaMemcpyAsync(tiled_prompt_lengths_buf_,
                        prefix_prompt_lengths.data(),
                        sizeof(int) * batch_size * beam_width,
                        cudaMemcpyDefault,
                        stream_);
    }

    sync_check_cuda_error();

    // handle first step
    if (has_prefix_prompt_ || has_prefix_soft_prompt_ || max_input_length > 1) {
        invokeTileGptInputs(tiled_input_ids_buf_,
                            tiled_input_lengths_buf_,
                            input_tensors->at("input_ids").getPtr<int>(),
                            input_tensors->at("input_lengths").getPtr<const int>(),
                            batch_size,
                            beam_width,
                            max_input_length,
                            stream_);
        sync_check_cuda_error();

        if (has_prefix_soft_prompt_) {
            inputIdsEmbeddingLookupPosEncodingSoftPromptParam<T> param;
            param.from_tensor                   = context_decoder_input_buf_;
            param.output_ids                    = output_ids_buf_;
            param.input_lengths                 = tiled_input_lengths_buf_;
            param.embedding_table               = gpt_weights->pre_decoder_embedding_table;
            param.pos_table                     = gpt_weights->position_encoding_table;
            param.prefix_soft_prompt_embedding  = input_tensors->at("request_prompt_embedding").getPtr<float>();
            param.prefix_soft_prompt_lengths    = input_tensors->at("request_prompt_lengths").getPtr<int>();
            param.input_ids                     = tiled_input_ids_buf_;
            param.start_step                    = 1;
            param.max_input_length              = max_input_length;
            param.max_prefix_soft_prompt_length = max_prefix_soft_prompt_length;
            param.batch_size                    = batch_size;
            param.beam_width                    = beam_width;
            param.hidden_units                  = hidden_units_;
            param.stream                        = stream_;

            invokeInputIdsEmbeddingLookupPosEncodingSoftPrompt(param);
            sync_check_cuda_error();
            max_input_length += max_prefix_soft_prompt_length;  // view soft_prompt as input
        }
        else {
            invokeInputIdsEmbeddingLookupPosEncoding(context_decoder_input_buf_,
                                                     output_ids_buf_,
                                                     gpt_weights->pre_decoder_embedding_table,
                                                     gpt_weights->position_encoding_table,
                                                     pPromptTuningParam<T>{},  // p/prompt tuning
                                                     tiled_input_ids_buf_,
                                                     1,
                                                     max_input_length,
                                                     max_input_length,
                                                     batch_size * beam_width,
                                                     hidden_units_,
                                                     stream_);
            sync_check_cuda_error();
        }
        
        invokeBuildDecoderAttentionMask(input_attention_mask_,
                                        tiled_input_lengths_buf_,
                                        tiled_prompt_lengths_buf_,
                                        batch_size * beam_width,
                                        max_input_length,
                                        max_prefix_prompt_length,
                                        stream_);
        sync_check_cuda_error();

        std::unordered_map<std::string, Tensor> decoder_input_tensors{
            {"decoder_input",
             Tensor{MEMORY_GPU,
                    data_type,
                    {batch_size * beam_width, (size_t)max_input_length, hidden_units_},
                    context_decoder_input_buf_}},
            {"attention_mask",
             Tensor{MEMORY_GPU,
                    data_type,
                    {batch_size * beam_width,
                     1,
                     (size_t)max_input_length,
                     (size_t)(max_input_length + max_prefix_prompt_length)},
                    input_attention_mask_}},
            {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, tiled_input_lengths_buf_}},
            {"d_prefix_prompt_batch",
             Tensor{MEMORY_GPU,
                    data_type,
                    {batch_size * beam_width},
                    has_prefix_prompt_ ? prompt_learning_weight_batch_ : nullptr}},
            {"d_prefix_prompt_lengths",
             Tensor{MEMORY_GPU,
                    TYPE_INT32,
                    {batch_size * beam_width},
                    has_prefix_prompt_ ? tiled_prompt_lengths_buf_ : nullptr}}};

        if (use_shared_contexts) {
            decoder_input_tensors.insert(
                {"compact_idx", Tensor(MEMORY_GPU, TYPE_INT32, {(size_t)compact_size}, compact_idx_)});
            decoder_input_tensors.insert(
                {"batch_to_compact_idx",
                 Tensor(MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, batch_to_compact_idx_)});
        }

        std::unordered_map<std::string, Tensor> decoder_output_tensors{
            {"decoder_output",
             Tensor{MEMORY_GPU,
                    data_type,
                    {batch_size * beam_width, (size_t)max_input_length, hidden_units_},
                    context_decoder_output_buf_}},
            {"key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_shape, key_cache_}},
            {"value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_shape, value_cache_}},
            {"last_token_hidden_units",
             Tensor{MEMORY_GPU, data_type, {batch_size * beam_width, hidden_units_}, decoder_output_buf_}}};
        
        forward(&decoder_output_tensors, &decoder_input_tensors, &gpt_weights->decoder_layer_weights);
            
        sync_check_cuda_error();
    }
    else if (max_input_length == 0) {
        FT_CHECK(prompt_learning_type_ == PromptLearningType::no_prompt
                 && request_prompt_type == PromptLearningType::no_prompt);  // Not support prompts in this case
        max_input_length++;
        invokeDecodingInitialize(finished_buf_,
                                 sequence_lengths_,
                                 output_ids_buf_,
                                 cum_log_probs_,
                                 start_ids_buf_,
                                 batch_size,
                                 beam_width,
                                 max_input_length - 1,
                                 stream_);
        std::vector<int> h_input_lengths(batch_size * beam_width, 1);
        cudaMemcpyAsync(tiled_input_lengths_buf_,
                        h_input_lengths.data(),
                        sizeof(int) * batch_size * beam_width,
                        cudaMemcpyHostToDevice,
                        stream_);
        sync_check_cuda_error();
    }
    else if (max_input_length == 1) {
        FT_CHECK(prompt_learning_type_ == PromptLearningType::no_prompt
                 && request_prompt_type == PromptLearningType::no_prompt);  // Not support prompts in this case
        invokeDecodingInitialize(finished_buf_,
                                 sequence_lengths_,
                                 nullptr,
                                 cum_log_probs_,
                                 start_ids_buf_,
                                 batch_size,
                                 beam_width,
                                 max_input_length - 1,
                                 stream_);
        sync_check_cuda_error();
        invokeTileGptInputs(tiled_input_ids_buf_,
                            tiled_input_lengths_buf_,
                            input_tensors->at("input_ids").getPtr<int>(),
                            input_tensors->at("input_lengths").getPtr<const int>(),
                            batch_size,
                            beam_width,
                            max_input_length,
                            stream_);
        sync_check_cuda_error();

        cudaMemcpyAsync(output_ids_buf_,
                        tiled_input_ids_buf_,
                        sizeof(int) * batch_size * beam_width,
                        cudaMemcpyDeviceToDevice,
                        stream_);
    }

    
}

template<typename T>
void LlamaPreprocessing<T>::forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                                    const std::unordered_map<std::string, Tensor>* input_tensors,
                                    const std::vector<LlamaDecoderLayerWeight<T>*>*  gpt_decoder_layer_weight)
{
    // input tensors:
    //      decoder_input [batch_size, seq_len, hidden_dimension],
    //      attention_mask [batch_size, 1, seq_len, seq_len + max_prompt_length]
    //      input_lengths [batch_size]
    //      d_prefix_prompt_batch [batch_size],
    //          each element contains ptr with buffer shape[2, local_head_num_, prompt_length, size_per_head]
    //      prefix_prompt_lengths [batch size]

    // output tensors:
    //      decoder_output [batch_size, seq_len, hidden_dimension],
    //      key_cache [num_layer, batch, local_head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [num_layer, batch, local_head_num, max_seq_len, size_per_head]
    //      last_token_hidden_units [batch_size, hidden_dimension]

    // To use layer/pipeline parallelism, we view the shape of 'batch_size' to 'ite * local_batch_size'.
    // For example, the shape of decoder_input becomes [ite, batch_size, seq_len, hidden_dimension] during
    // computing.

    FT_CHECK(input_tensors->size() >= 5);
    FT_CHECK(output_tensors->size() == 4);

    const bool use_shared_contexts = input_tensors->find("compact_idx") != input_tensors->end();
    FT_CHECK(!use_shared_contexts || (input_tensors->find("batch_to_compact_idx") != input_tensors->end()));
    const size_t request_batch_size = input_tensors->at("decoder_input").shape[0];
    // compacted batch size.
    const size_t batch_size =
        use_shared_contexts ? input_tensors->at("compact_idx").shape[0] : input_tensors->at("decoder_input").shape[0];

    const int seq_len    = input_tensors->at("decoder_input").shape[1];  // max_input_len
    // The maximum length of generation.
    const size_t max_seq_len = output_tensors->at("value_cache").shape[3];

    const int max_prompt_length =
        input_tensors->at("attention_mask").shape[3] - input_tensors->at("attention_mask").shape[2];
    const DataType data_type = getTensorType<T>();
    
    T*         decoder_input           = input_tensors->at("decoder_input").getPtr<T>();
    T*         decoder_output          = output_tensors->at("decoder_output").getPtr<T>();
    const T*   attention_mask          = input_tensors->at("attention_mask").getPtr<const T>();
    const T**  d_prefix_prompt_batch   = input_tensors->at("d_prefix_prompt_batch").getPtr<const T*>();
    const int* d_prefix_prompt_lengths = input_tensors->at("d_prefix_prompt_lengths").getPtr<const int>();

    if (use_shared_contexts) {
        invokeCompactInputs(compact_decoder_features_,
                            compact_attention_mask_,
                            compact_input_lengths_,
                            decoder_input,
                            attention_mask,
                            input_tensors->at("input_lengths").getPtr<int>(),
                            input_tensors->at("compact_idx").getPtr<int>(),
                            batch_size,
                            seq_len,
                            hidden_units_,
                            stream_);
    }

    const int local_batch_size = getLocalBatchSize(batch_size, seq_len, pipeline_para_.world_size_);
    FT_CHECK(batch_size % local_batch_size == 0);
    const int iteration_num = batch_size / local_batch_size;

    Tensor&             k_cache = output_tensors->at("key_cache");
    Tensor&             v_cache = output_tensors->at("value_cache");
    std::vector<size_t> self_k_cache_size;
    self_k_cache_size.push_back(local_batch_size);
    for (auto t = k_cache.shape.begin() + 2; t != k_cache.shape.end(); ++t) {
        self_k_cache_size.push_back(*t);
    }
    std::vector<size_t> self_v_cache_size;
    self_v_cache_size.push_back(local_batch_size);
    for (auto t = v_cache.shape.begin() + 2; t != v_cache.shape.end(); ++t) {
        self_v_cache_size.push_back(*t);
    }

    AttentionType attention_type  = (d_prefix_prompt_lengths != nullptr) ?
                                        getUnfusedAttentionType(attention_type_) :
                                        attention_type_;
    const bool    is_unpadded_mha = isUnPaddedMHA(attention_type);
    
    for (int ite = 0; ite < iteration_num; ite++) {
        size_t h_token_num = local_batch_size * seq_len;
        if (is_unpadded_mha) {
            const int* base_input_lengths =
                use_shared_contexts ? compact_input_lengths_ : input_tensors->at("input_lengths").getPtr<int>();
            fprintf(stdout, "token num %d, seq_len %d\n", h_token_num, seq_len);
            invokeGetPaddingOffsetAndCuSeqLens(h_pinned_token_num_ptr_,
                                               &h_token_num,
                                               padding_offset_,
                                               cu_seqlens_,
                                               base_input_lengths + ite * local_batch_size,
                                               local_batch_size,
                                               seq_len,
                                               stream_);
        }
        for (int l = 0; l < num_layer_; l++) {
            if (isValidLayerParallelId(l) == false) {
                continue;
            }

            if (l == 0 && is_unpadded_mha) {
                const T* base_input = (use_shared_contexts ? compact_decoder_features_ : decoder_input);
                invokeRemovePadding(decoder_layer_output_,
                                    base_input + ite * local_batch_size * seq_len * hidden_units_,
                                    padding_offset_,
                                    h_token_num,
                                    hidden_units_,
                                    stream_);
            }

            const bool is_final     = false;  // TODO(bhsueh) remove this flag
            T* layer_input  = decoder_layer_output_;
            T* layer_output = decoder_layer_output_;
            if (!is_unpadded_mha) {
                if (l == 0) {
                    layer_input = use_shared_contexts ? compact_decoder_features_ : decoder_input;
                    layer_input += ite * local_batch_size * seq_len * hidden_units_;
                }
                if (l == num_layer_ - 1) {
                    layer_output = use_shared_contexts ? compact_decoder_features_ : decoder_output;
                    layer_output += ite * local_batch_size * seq_len * hidden_units_;
                }
            }
            
            if (isFirstLayerParallelId(l) && pipeline_para_.rank_ != 0 && pipeline_para_.world_size_ > 1) {
                int data_size = h_token_num * hidden_units_ / tensor_para_.world_size_;
                ftNcclRecv(layer_input + data_size * tensor_para_.rank_,
                           data_size,
                           pipeline_para_.rank_ - 1,
                           pipeline_para_,
                           stream_);
                           
                if (tensor_para_.world_size_ > 1) {
                    ftNcclAllGather(layer_input, layer_input, data_size, tensor_para_.rank_, tensor_para_, stream_);
                }
            }
            
            // TODO: 这里用的LN跟neox不一样，不太清楚这里需不需要改成int8的LN
            invokeGeneralT5LayerNorm(decoder_normed_input_,
                                   layer_input,
                                   gpt_decoder_layer_weight->at(l)->pre_layernorm_weights.gamma,
                                   (const T*)nullptr,
                                   layernorm_eps_,
                                   h_token_num,
                                   hidden_units_,
                                   stream_);

            sync_check_cuda_error();

            const T* attention_ptr = use_shared_contexts ? compact_attention_mask_ : attention_mask;

            TensorMap self_attention_input_tensors{
                {"input_query",
                 Tensor{MEMORY_GPU, data_type, {h_token_num, (size_t)hidden_units_}, decoder_normed_input_}},
                {"attention_mask",
                 Tensor{MEMORY_GPU,
                        data_type,
                        {(size_t)local_batch_size, (size_t)1, (size_t)seq_len, (size_t)(seq_len + max_prompt_length)},
                        attention_ptr + local_batch_size * ite * seq_len * (seq_len + max_prompt_length)}},
                {"attention_type", Tensor{MEMORY_CPU, TYPE_VOID, {1}, &attention_type}},
                {"is_final_layer", Tensor{MEMORY_CPU, TYPE_BOOL, {(size_t)1}, &is_final}},
                {"layer_id", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &l}}};

            self_attention_input_tensors.insertIfValid(
                "d_prefix_prompt_batch",
                Tensor{MEMORY_GPU,
                       data_type,
                       {(size_t)local_batch_size},
                       d_prefix_prompt_batch != nullptr ? d_prefix_prompt_batch + ite * local_batch_size : nullptr});
            self_attention_input_tensors.insertIfValid("d_prefix_prompt_lengths",
                                                       Tensor{MEMORY_GPU,
                                                              TYPE_INT32,
                                                              {(size_t)local_batch_size},
                                                              d_prefix_prompt_lengths != nullptr ?
                                                                  d_prefix_prompt_lengths + ite * local_batch_size :
                                                                  nullptr});
            if (is_unpadded_mha) {
                self_attention_input_tensors.insert("padding_offset",
                                                    Tensor{MEMORY_GPU, TYPE_INT32, {h_token_num}, padding_offset_});
                self_attention_input_tensors.insert(
                    "cu_seqlens", Tensor{MEMORY_GPU, TYPE_INT32, {size_t(local_batch_size + 1)}, cu_seqlens_});
            }

            // NOTE: cache offer for specific layer
            size_t cache_offset = l - getFirstLayerParallelId();
            for (auto t = k_cache.shape.begin() + 1; t != k_cache.shape.end(); ++t) {
                cache_offset *= *t;
            };
            size_t ite_cache_offset = ite * local_batch_size;
            for (auto t = k_cache.shape.begin() + 2; t != k_cache.shape.end(); ++t) {
                ite_cache_offset *= *t;
            }
            cache_offset += ite_cache_offset;

            T* k_cache_ptr = use_shared_contexts ? k_cache_layer_ : k_cache.getPtrWithOffset<T>(cache_offset);
            T* v_cache_ptr = use_shared_contexts ? v_cache_layer_ : v_cache.getPtrWithOffset<T>(cache_offset);

            TensorMap self_attention_output_tensors{
                {"hidden_features",
                 Tensor{MEMORY_GPU, data_type, {h_token_num, (size_t)hidden_units_}, self_attn_output_}},
                {"key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_size, k_cache_ptr}},
                {"value_cache",
                 Tensor{MEMORY_GPU, data_type, self_v_cache_size, v_cache_ptr}}};

            self_attention_layer_->forward(&self_attention_output_tensors,
                                           &self_attention_input_tensors,
                                           &gpt_decoder_layer_weight->at(l)->self_attention_weights);

            if (use_shared_contexts) {
                // Even with local batches, we must process the whole K/V caches as any
                // element in batch_idx_to_compact_idx may reference the local batch
                // we're processing. We also need to discard references that aren't in
                // that particular local batch.
                const size_t cache_stride_per_batch = hidden_units_ / tensor_para_.world_size_ * max_seq_len;
                const size_t cache_layer_offset =
                    (l - getFirstLayerParallelId()) * request_batch_size * cache_stride_per_batch;
                invokeUnCompactCaches(k_cache.getPtrWithOffset<T>(cache_layer_offset),
                                      v_cache.getPtrWithOffset<T>(cache_layer_offset),
                                      k_cache_layer_,
                                      v_cache_layer_,
                                      input_tensors->at("batch_to_compact_idx").getPtr<int>(),
                                      request_batch_size,  // batch_size (uncompact)
                                      v_cache.shape[2],    // local_head_num
                                      max_seq_len,
                                      seq_len,
                                      size_per_head_,
                                      local_batch_size,
                                      ite,
                                      stream_);
                sync_check_cuda_error();
            }

            if (is_final == false) {
                if (use_gptj_residual_) {
                    invokeGeneralLayerNorm(decoder_normed_input_,
                                           layer_input,
                                           gpt_decoder_layer_weight->at(l)->post_attention_layernorm_weights.gamma,
                                           gpt_decoder_layer_weight->at(l)->post_attention_layernorm_weights.beta,
                                           layernorm_eps_,
                                           h_token_num,
                                           hidden_units_,
                                           (float*)nullptr,
                                           int8_mode_,
                                           stream_);
                }
                else {
                    // TODO: modify or not ?
                    invokeGeneralAddResidualT5PreLayerNorm(
                        self_attn_output_,
                        decoder_normed_input_,
                        layer_input,
                        gpt_decoder_layer_weight->at(l)->post_attention_layernorm_weights.gamma,
                        layernorm_eps_,
                        h_token_num,
                        hidden_units_,
                        stream_);
                }

                TensorMap ffn_input_tensors(
                    {{"ffn_input",
                      Tensor{MEMORY_GPU, data_type, {h_token_num, (size_t)hidden_units_}, decoder_normed_input_}}});
                TensorMap ffn_output_tensors(
                    {{"ffn_output", Tensor{MEMORY_GPU, data_type, {h_token_num, (size_t)hidden_units_}, use_gptj_residual_ ? ffn_output_ : layer_output}}});
                ffn_layer_->forward(
                    &ffn_output_tensors, &ffn_input_tensors, &gpt_decoder_layer_weight->at(l)->ffn_weights);

                if (use_gptj_residual_) {
                    // Original workflow:
                    //      layer_output = layer_input + reduceSum(ffn_output + self_attn_output + ffn_output_bias)
                    // Our workflow:
                    //      layer_output = reduceSum(ffn_output + self_attn_output + ffn_output_bias + layer_input /
                    //      TP_size)
                    // They are equivalent on math, but we can use same buffer for layer_input and layer_output

                    invokeAddBiasAttentionFfnResidual(layer_output,
                                                      ffn_output_,
                                                      self_attn_output_,
                                                      layer_input,
                                                      gpt_decoder_layer_weight->at(l)->ffn_weights.output_weight.bias,
                                                      h_token_num,
                                                      hidden_units_,
                                                      tensor_para_.world_size_,
                                                      stream_);
                    if (tensor_para_.world_size_ > 1) {
                        ftNcclAllReduceSum(
                            layer_output, layer_output, h_token_num * hidden_units_, tensor_para_, stream_);
                    }
                }
                else {
                    invokeAddBiasResidual(layer_output,
                                          self_attn_output_,
                                          gpt_decoder_layer_weight->at(l)->ffn_weights.output_weight.bias,
                                          h_token_num,
                                          hidden_units_,
                                          stream_);
                }
                sync_check_cuda_error();

                if (isLastLayerParallelId(l) && pipeline_para_.rank_ != pipeline_para_.world_size_ - 1
                    && pipeline_para_.world_size_ > 1) {
                    int data_size = h_token_num * hidden_units_ / tensor_para_.world_size_;
                    ftNcclSend(layer_output + data_size * tensor_para_.rank_,
                               data_size,
                               pipeline_para_.rank_ + 1,
                               pipeline_para_,
                               stream_);
                }

                if ((l == num_layer_ - 1) && is_unpadded_mha) {
                    T* base_ptr = use_shared_contexts ? compact_decoder_features_ : decoder_output;
                    invokeRebuildPadding(base_ptr + ite * local_batch_size * seq_len * hidden_units_,
                                         decoder_layer_output_,
                                         padding_offset_,
                                         h_token_num,
                                         head_num_ * size_per_head_,
                                         stream_);
                }
            }
        }
    }

    
    // TODO(bhsueh) We could optimize this point by only computing the last token for the last layer
    invokeLookupHiddenStateOfLastToken(output_tensors->at("last_token_hidden_units").getPtr<T>(),
                                       output_tensors->at("decoder_output").getPtr<T>(),
                                       input_tensors->at("input_lengths").getPtr<int>(),
                                       seq_len,
                                       batch_size,
                                       hidden_units_,
                                       stream_);
    sync_check_cuda_error();
}


template class LlamaPreprocessing<float>;
template class LlamaPreprocessing<half>;
#ifdef ENABLE_BF16
template class LlamaPreprocessing<__nv_bfloat16>;
#endif
}