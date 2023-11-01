
#include "gptj_inference.h"
#include "src/flover/layers/TensorParallelGeluFfnLayer.h"
#include "src/flover/layers/attention_layers/TensorParallelDecoderSelfAttentionLayer.h"
#include "src/flover/kernels/bert_preprocess_kernels.h"
#include "src/flover/kernels/decoding_kernels.h"
#include "src/flover/kernels/gpt_kernels.h"
#include "src/flover/layers/beam_search_layers/BaseBeamSearchLayer.h"
#include <algorithm>


namespace flover {

template<typename T>
void GptJInference<T>::initialize()
{
    self_attention_layer_ = new TensorParallelDecoderSelfAttentionLayer<T>(max_batch_size_,
                                                                           head_num_,
                                                                           size_per_head_,
                                                                           rotary_embedding_dim_,
                                                                           neox_rotary_style_,
                                                                           tensor_para_,
                                                                           stream_,
                                                                           cublas_wrapper_,
                                                                           nullptr,
                                                                           true,
                                                                           is_free_buffer_after_forward_,
                                                                           false,
                                                                           0,
                                                                           custom_all_reduce_comm_,
                                                                           enable_custom_all_reduce_);

    ffn_layer_ = new TensorParallelGeluFfnLayer<T>(max_batch_size_,
                                                   1,
                                                   head_num_,
                                                   size_per_head_,
                                                   0,  // expert_num
                                                   inter_size_,
                                                   tensor_para_,
                                                   stream_,
                                                   cublas_wrapper_,
                                                   nullptr,
                                                   true,
                                                   is_free_buffer_after_forward_,
                                                   false,
                                                   0,
                                                   false,  // use_gated_activation = false;
                                                   custom_all_reduce_comm_,
                                                   enable_custom_all_reduce_);
}

template<typename T>
void GptJInference<T>::allocateBuffer()
{
}

template<typename T>
void GptJInference<T>::freeBuffer()
{
}


template<typename T>
bool GptJInference<T>::isValidLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T>
bool GptJInference<T>::isFirstLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T>
bool GptJInference<T>::isLastLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T>
int GptJInference<T>::getFirstLayerParallelId()
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T>
GptJInference<T>::GptJInference(size_t                              max_batch_size,
                            size_t                              head_num,
                            size_t                              size_per_head,
                            size_t                              inter_size,
                            size_t                              num_layer,
                            int                                 vocab_size,
                            size_t                              rotary_embedding_dim,
                            bool                                neox_rotary_style,
                            float                               layernorm_eps,
                            NcclParam                           tensor_para,
                            NcclParam                           pipeline_para,
                            cudaStream_t                        stream,
                            cublasMMWrapper*                    cublas_wrapper,
                            bool                                is_free_buffer_after_forward,
                            std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                            int                                 enable_custom_all_reduce):
    BaseLayer(stream, cublas_wrapper, nullptr, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    rotary_embedding_dim_(rotary_embedding_dim),
    neox_rotary_style_(neox_rotary_style),
    layernorm_eps_(layernorm_eps),
    hidden_units_(head_num_ * size_per_head),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    initialize();
}

template<typename T>
GptJInference<T>::~GptJInference()
{
    delete self_attention_layer_;
    delete ffn_layer_;
}


template<typename T>
void GptJInference<T>::run(int* stop_flag, 
                           const INIReader& reader, 
                           const GptJWeight<T>* gpt_weights, 
                           InferenceQueue* inque)
{

    const std::string model_name               = reader.Get("model_specification", "model_type");
    const size_t      max_concurrency          = reader.GetInteger("model_specification", "max_concurrency");
    const size_t      per_batch_size           = reader.GetInteger("model_specification", "per_batch_size");
    const size_t      total_seq_len            = reader.GetInteger("model_specification", "max_seq_len");
    const size_t      beam_width               = reader.GetInteger("model_specification", "beam_width");
    const size_t      batch_size               = per_batch_size * max_concurrency;
    const int         fixed_input_len          = reader.GetInteger("model_specification", "fixed_input_len");
    const int         fixed_prompt_len         = reader.GetInteger("model_specification", "fixed_prompt_len");
    const int         start_id                 = reader.GetInteger(model_name, "start_id");
    const int         end_id                   = reader.GetInteger(model_name, "end_id");
    const size_t      head_num                 = reader.GetInteger(model_name, "head_num");
    const size_t      size_per_head            = reader.GetInteger(model_name, "size_per_head");

    const size_t      hidden_units             = head_num * size_per_head;
    bool              has_prefix_prompt_       = false;
    size_t            max_prefix_prompt_length = fixed_prompt_len;
    const size_t      max_cache_seq_len        = total_seq_len + max_prefix_prompt_length;
    int               step                     = max_cache_seq_len;
    int               ite                      = 0;

    const size_t      local_head_num_      = head_num_ / tensor_para_.world_size_;
    std::vector<size_t> self_k_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                batch_size * beam_width,
                                                local_head_num_,
                                                size_per_head_ / (16 / sizeof(T)),
                                                max_cache_seq_len,
                                                16 / sizeof(T)};
    std::vector<size_t> self_v_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                batch_size * beam_width,
                                                local_head_num_,
                                                max_cache_seq_len,
                                                size_per_head_};

    int local_vacab_size = ceil(vocab_size_ / 1.f / tensor_para_.world_size_);
    if (std::is_same<half, T>::value) {
        local_vacab_size = ceil(local_vacab_size / 8.f) * 8;
    }
    size_t vocab_size_padded_ = (size_t)local_vacab_size * tensor_para_.world_size_;
    int max_output_seq_len = total_seq_len;
    int max_input_length   = fixed_input_len;
    
    const DataType       data_type      = getTensorType<T>();
    const cudaDataType_t gemm_data_type = getCudaDataType<T>();
    
    total_iters = fixed_input_len;
    while (true) {
        InferenceInfo _req;
        if (inque->get(_req)) {
            ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
            log("Rank ", tensor_para_.rank_, " :: -- inference task id ", _req.unique_task_id);
            // i.decoder_input_buf_len         = batchxbeam * hidden_units;
            // i.output_ids_buf_len            = batchxbeam * total_seq_len;
            // i.tiled_total_padding_count_len = batchxbeam;
            // i.finished_buf_len              = batchxbeam;
            // i.sequence_lengths_len          = batchxbeam;
            // i.tiled_prompt_lengths_buf_len  = batchxbeam;
            // i.cache_indirections_len        = batchxbeam * max_cache_seq_len;
            // i.masked_tokens_len             = batchxbeam * max_cache_seq_len;
            // i.decoder_output_buf_len        = batchxbeam * hidden_units;
            // i.key_cache_len                 = batchxbeam * max_cache_seq_len;
            // i.value_cache_len               = batchxbeam * max_cache_seq_len;

            if (_req.decoder_input_buf_len > 0){
                actives += 1;
                running_batchsize             += per_batch_size;
                decoder_input_buf_len         += _req.decoder_input_buf_len;
                output_ids_buf_len            += _req.output_ids_buf_len;
                tiled_total_padding_count_len += _req.tiled_total_padding_count_len;
                finished_buf_len              += _req.finished_buf_len;
                sequence_lengths_len          += _req.sequence_lengths_len;
                tiled_prompt_lengths_buf_len  += _req.tiled_prompt_lengths_buf_len;
                cache_indirections_len        += _req.cache_indirections_len;
                masked_tokens_len             += _req.masked_tokens_len;
                decoder_output_buf_len        += _req.decoder_output_buf_len;
                key_cache_len                 += _req.key_cache_len;
                value_cache_len               += _req.value_cache_len;
                
                InferenceInfo* heap_req       = new InferenceInfo(_req);
                std::unique_ptr<InferenceInfo> ptr(heap_req);

                InferenceStatus[_req.unique_task_id] = std::move(ptr);

                self_k_cache_shape.at(1) = running_batchsize * beam_width;
                self_v_cache_shape.at(1) = running_batchsize * beam_width;

                sync_check_cuda_error();

                invokeDecodingInitialize(finished_buf_ + _req.mem_id * _req.finished_buf_len,
                                        sequence_lengths_ + _req.mem_id * _req.sequence_lengths_len,
                                        nullptr,
                                        cum_log_probs_ + _req.mem_id * _req.cum_log_probs_len,
                                        start_ids_buf_ + _req.mem_id * _req.start_ids_buf_len,
                                        per_batch_size,
                                        beam_width,
                                        max_input_length - 1,
                                        stream_);
                sync_check_cuda_error();

                invokeMaskPaddingTokens(masked_tokens_ + _req.mem_id * _req.masked_tokens_len,
                                        _req.d_input_lengths,  // not_tiled
                                        tiled_prompt_lengths_buf_ + _req.mem_id * _req.tiled_prompt_lengths_buf_len,
                                        max_cache_seq_len,
                                        max_input_length + max_prefix_prompt_length,
                                        0,
                                        per_batch_size,
                                        beam_width,
                                        stream_);
                sync_check_cuda_error();
                ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
                log("Rank ", tensor_para_.rank_, " Inference request", _req.unique_task_id, " prepare done ");
            }
        }
        if (actives) {
            ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
            const int src_indir_idx = (total_iters - max_input_length) % 2;
            const int tgt_indir_idx = 1 - src_indir_idx;

            for (auto& pair : InferenceStatus) {
                int key = pair.first;
                InferenceInfo* value = pair.second.get();
                if(value->cur_step > max_input_length){
                    invokeEmbeddingLookupPosEncodingPadCount(decoder_input_buf_ + value->mem_id * value->decoder_input_buf_len,
                                                             gpt_weights->pre_decoder_embedding_table,
                                                             gpt_weights->position_encoding_table,
                                                             output_ids_buf_ + value->mem_id * value->output_ids_buf_len,
                                                             tiled_total_padding_count_ + value->mem_id * value->tiled_total_padding_count_len,
                                                             per_batch_size * beam_width,
                                                             hidden_units,
                                                             (T)(1.0f),
                                                             value->cur_step - 1,
                                                             per_batch_size * beam_width,
                                                             0,
                                                             stream_);
                    sync_check_cuda_error();
                }
            }

            std::unordered_map<std::string, Tensor> decoder_input_tensors{
                {"decoder_input",
                    Tensor{MEMORY_GPU,
                        data_type,
                        {running_batchsize * beam_width, hidden_units},
                        decoder_input_buf_ + decoder_input_buf_offset}},
                {"finished",
                    Tensor{MEMORY_GPU, TYPE_BOOL, {running_batchsize * beam_width}, finished_buf_ + finished_buf_offset}},
                {"sequence_lengths",
                    Tensor{MEMORY_GPU, TYPE_INT32, {running_batchsize * beam_width}, sequence_lengths_ + sequence_lengths_offset}},
                {"total_padding_tokens",
                    Tensor{MEMORY_GPU,
                        TYPE_INT32,
                        {running_batchsize * beam_width},
                        tiled_total_padding_count_ + tiled_total_padding_count_offset}},
                {"d_prefix_prompt_lengths",
                    Tensor{MEMORY_GPU,
                        TYPE_INT32,
                        {running_batchsize},
                        has_prefix_prompt_ ? (tiled_prompt_lengths_buf_ + tiled_prompt_lengths_buf_offset) : nullptr}},
                {"max_prefix_prompt_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_prefix_prompt_length}},
                {"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length}},
                {"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step}},
                {"ite", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &ite}},
                {"cache_indirection",
                    Tensor{MEMORY_GPU,
                        TYPE_INT32,
                        {running_batchsize, beam_width, max_output_seq_len},
                        beam_width > 1 ? cache_indirections_[src_indir_idx] + cache_indirections_offset :
                                            nullptr}},
                {"masked_tokens",
                    Tensor{MEMORY_GPU,
                        TYPE_BOOL,
                        {running_batchsize * beam_width, max_cache_seq_len},
                        masked_tokens_ + masked_tokens_offset}}};
            std::unordered_map<std::string, Tensor> decoder_output_tensors{
                {"decoder_output",
                    Tensor{MEMORY_GPU,
                        data_type,
                        {running_batchsize * beam_width, hidden_units},
                        decoder_output_buf_ + decoder_output_buf_offset}},
                {"key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_shape, key_cache_ + key_cache_offset}},
                {"value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_shape, value_cache_ + value_cache_offset}}};
            
            forward(&decoder_output_tensors, &decoder_input_tensors, &gpt_weights->decoder_layer_weights);
            
            total_iters += 1;
            log("Rank ", tensor_para_.rank_, " Total iters ", total_iters, ", actives ", actives, ", working batch size ", running_batchsize * beam_width, ", cache ind ", src_indir_idx);

            for (auto it = InferenceStatus.begin(); it != InferenceStatus.end(); /* no increment here */) {
                int key              = it->first;
                InferenceInfo* value = it->second.get();
                value->cur_step      += 1;
                if(value->cur_step >= value->total_seq_len){
                    actives                          -=1;
                    running_batchsize                -= per_batch_size;
                    decoder_input_buf_offset         += _req.decoder_input_buf_len;
                    output_ids_buf_offset            += _req.output_ids_buf_len;
                    tiled_total_padding_count_offset += _req.tiled_total_padding_count_len;
                    finished_buf_offset              += _req.finished_buf_len;
                    sequence_lengths_offset          += _req.sequence_lengths_len;
                    tiled_prompt_lengths_buf_offset  += _req.tiled_prompt_lengths_buf_len;
                    cache_indirections_offset        += _req.cache_indirections_len;
                    masked_tokens_offset             += _req.masked_tokens_len;
                    decoder_output_buf_offset        += _req.decoder_output_buf_len;
                    key_cache_offset                 += _req.key_cache_len;
                    value_cache_offset               += _req.value_cache_len;
                    InferenceStatus.erase(it);
                    log("Rank ", tensor_para_.rank_, " task ", key, " finished");
                    ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
                    break;
                }else {
                    ++it;
                }
            }
            ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);

        } else {
            if (*stop_flag == 1) {
                InferenceStatus.clear();
                break;
            }
        }
    }



    // for (int step = max_input_length; step < (int)max_output_seq_len; step++) {
    //     fprintf(stdout, "%d\n", step);
    //     const int src_indir_idx = (step - max_input_length) % 2;
    //     const int tgt_indir_idx = 1 - src_indir_idx;

    //     // *generation_should_stop_   = true;

    //     const size_t local_batch_size = getLocalBatchSize(batch_size, 1, pipeline_para_.world_size_);
    //     FT_CHECK(batch_size % local_batch_size == 0);
    //     const size_t iteration_num = batch_size / local_batch_size;

    //     for (uint ite = 0; ite < iteration_num; ++ite) {
    //         const int id_offset               = ite * local_batch_size * beam_width;
    //         const int hidden_units_offset     = id_offset * hidden_units_;
    //         const int vocab_size_units_offset = id_offset * vocab_size_padded_;

    //         if (!(max_input_length > 1 && step == max_input_length)) {
    //             if (pipeline_para_.rank_ == 0) {
    //                 // invokeEmbeddingLookupPosEncodingPadCount(decoder_input_buf_ + hidden_units_offset,
    //                 //                                          gpt_weights->pre_decoder_embedding_table,
    //                 //                                          gpt_weights->position_encoding_table,
    //                 //                                          output_ids_buf_ + id_offset,
    //                 //                                          tiled_total_padding_count_ + id_offset,
    //                 //                                          local_batch_size * beam_width,
    //                 //                                          hidden_units_,
    //                 //                                          (T)(1.0f),
    //                 //                                          step - 1,
    //                 //                                          batch_size * beam_width,
    //                 //                                          0,
    //                 //                                          stream_);
    //                 sync_check_cuda_error();
    //             }
                
    //             // std::unordered_map<std::string, Tensor> decoder_input_tensors{
    //             //     {"decoder_input",
    //             //      Tensor{MEMORY_GPU,
    //             //             data_type,
    //             //             {local_batch_size * beam_width, hidden_units_},
    //             //             decoder_input_buf_ + hidden_units_offset}},
    //             //     {"finished",
    //             //      Tensor{MEMORY_GPU, TYPE_BOOL, {local_batch_size * beam_width}, finished_buf_ + id_offset}},
    //             //     {"sequence_lengths",
    //             //      Tensor{MEMORY_GPU, TYPE_INT32, {local_batch_size * beam_width}, sequence_lengths_ + id_offset}},
    //             //     {"total_padding_tokens",
    //             //      Tensor{MEMORY_GPU,
    //             //             TYPE_INT32,
    //             //             {local_batch_size * beam_width},
    //             //             tiled_total_padding_count_ + id_offset}},
    //             //     {"d_prefix_prompt_lengths",
    //             //      Tensor{MEMORY_GPU,
    //             //             TYPE_INT32,
    //             //             {local_batch_size},
    //             //             has_prefix_prompt_ ? (tiled_prompt_lengths_buf_ + id_offset) : nullptr}},
    //             //     {"max_prefix_prompt_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_prefix_prompt_length}},
    //             //     {"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length}},
    //             //     {"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step}},
    //             //     {"ite", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &ite}},
    //             //     {"cache_indirection",
    //             //      Tensor{MEMORY_GPU,
    //             //             TYPE_INT32,
    //             //             {local_batch_size, beam_width, max_output_seq_len},
    //             //             beam_width > 1 ? cache_indirections_[src_indir_idx] + id_offset * max_output_seq_len :
    //             //                              nullptr}},
    //             //     {"masked_tokens",
    //             //      Tensor{MEMORY_GPU,
    //             //             TYPE_BOOL,
    //             //             {local_batch_size * beam_width, max_cache_seq_len},
    //             //             masked_tokens_ + id_offset * max_cache_seq_len}}};
    //             // std::unordered_map<std::string, Tensor> decoder_output_tensors{
    //             //     {"decoder_output",
    //             //      Tensor{MEMORY_GPU,
    //             //             data_type,
    //             //             {local_batch_size * beam_width, hidden_units_},
    //             //             decoder_output_buf_ + hidden_units_offset}},
    //             //     {"key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_shape, key_cache_}},
    //             //     {"value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_shape, value_cache_}}};
                    
    //             // forward(&decoder_output_tensors, &decoder_input_tensors, &gpt_weights->decoder_layer_weights);
    //         }

    //         // if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
    //         //     invokeGeneralLayerNorm(normed_decoder_output_buf_ + hidden_units_offset,
    //         //                            decoder_output_buf_ + hidden_units_offset,
    //         //                            gpt_weights->post_decoder_layernorm.gamma,
    //         //                            gpt_weights->post_decoder_layernorm.beta,
    //         //                            layernorm_eps_,
    //         //                            local_batch_size * beam_width,
    //         //                            hidden_units_,
    //         //                            (float*)nullptr,
    //         //                            0,
    //         //                            stream_);
    //         //     sync_check_cuda_error();

    //         //     if (tensor_para_.world_size_ == 1) {
    //         //         float alpha = 1.0f;
    //         //         float beta  = 0.0f;
    //         //         cublas_wrapper_->Gemm(CUBLAS_OP_T,
    //         //                               CUBLAS_OP_N,
    //         //                               vocab_size_padded_,  // n
    //         //                               local_batch_size * beam_width,
    //         //                               hidden_units_,  // k
    //         //                               &alpha,
    //         //                               padded_embedding_kernel_ptr_,
    //         //                               gemm_data_type,
    //         //                               hidden_units_,  // k
    //         //                               normed_decoder_output_buf_ + hidden_units_offset,
    //         //                               gemm_data_type,
    //         //                               hidden_units_,  // k
    //         //                               &beta,
    //         //                               logits_buf_ + vocab_size_units_offset,
    //         //                               CUDA_R_32F,
    //         //                               vocab_size_padded_, /* n */
    //         //                               CUDA_R_32F,
    //         //                               cublasGemmAlgo_t(-1));
    //         //     }
    //         //     else {
    //         //         FT_CHECK(vocab_size_padded_ % tensor_para_.world_size_ == 0);
    //         //         const int local_vocab_size = vocab_size_padded_ / tensor_para_.world_size_;
    //         //         float     alpha            = 1.0f;
    //         //         float     beta             = 0.0f;

    //         //         cublas_wrapper_->Gemm(CUBLAS_OP_T,
    //         //                               CUBLAS_OP_N,
    //         //                               local_vocab_size,  // n
    //         //                               local_batch_size * beam_width,
    //         //                               hidden_units_,  // k
    //         //                               &alpha,
    //         //                               padded_embedding_kernel_ptr_
    //         //                                   + tensor_para_.rank_ * local_vocab_size * hidden_units_,
    //         //                               gemm_data_type,
    //         //                               hidden_units_,  // k
    //         //                               normed_decoder_output_buf_ + hidden_units_offset,
    //         //                               gemm_data_type,
    //         //                               hidden_units_,  // k
    //         //                               &beta,
    //         //                               nccl_logits_buf_ + vocab_size_units_offset
    //         //                                   + tensor_para_.rank_ * local_batch_size * beam_width * local_vocab_size,
    //         //                               CUDA_R_32F,
    //         //                               local_vocab_size, /* n */
    //         //                               CUDA_R_32F,
    //         //                               cublasGemmAlgo_t(-1));
                                          
    //         //         ftNcclAllGather(nccl_logits_buf_ + vocab_size_units_offset,
    //         //                         nccl_logits_buf_ + vocab_size_units_offset,
    //         //                         local_batch_size * beam_width * local_vocab_size,
    //         //                         tensor_para_.rank_,
    //         //                         tensor_para_,
    //         //                         stream_);
                    
    //         //         invokeTransposeAxis01(logits_buf_ + vocab_size_units_offset,
    //         //                               nccl_logits_buf_ + vocab_size_units_offset,
    //         //                               tensor_para_.world_size_,
    //         //                               local_batch_size * beam_width,
    //         //                               local_vocab_size,
    //         //                               stream_);

    //         //     }


    //         //     invokeGenericActivation<IdentityActivation, float, T>(logits_buf_ + vocab_size_units_offset,
    //         //                                                           padded_embedding_bias_ptr_,
    //         //                                                           nullptr,
    //         //                                                           nullptr,
    //         //                                                           nullptr,
    //         //                                                           nullptr,
    //         //                                                           local_batch_size * beam_width,
    //         //                                                           vocab_size_padded_,
    //         //                                                           0,
    //         //                                                           nullptr,
    //         //                                                           nullptr,
    //         //                                                           stream_);

    //         //     // int                                     tmp_local_batch_size       = local_batch_size;
    //         //     // bool                                    is_initialize_random_table = false;
    //         //     // std::unordered_map<std::string, Tensor> dynamic_decode_input_tensors{
    //         //     //     {"logits",
    //         //     //      Tensor{MEMORY_GPU, TYPE_FP32, {batch_size, beam_width, vocab_size_padded_}, logits_buf_}},
    //         //     //     // {"embedding_bias", Tensor{MEMORY_GPU, data_type, {vocab_size_padded_}, nullptr}},
    //         //     //     {"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step}},
    //         //     //     {"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length}},
    //         //     //     {"input_lengths",
    //         //     //      Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width}, tiled_input_lengths_buf_}},
    //         //     //     {"sequence_limit_length", Tensor{MEMORY_GPU, TYPE_UINT32, {batch_size}, seq_limit_len_}},
    //         //     //     {"ite", Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &ite}},
    //         //     //     {"src_cache_indirection",
    //         //     //      Tensor{MEMORY_GPU,
    //         //     //             TYPE_INT32,
    //         //     //             {local_batch_size, beam_width, max_output_seq_len},
    //         //     //             cache_indirections_[src_indir_idx] + id_offset * max_output_seq_len}},
    //         //     //     {"local_batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &tmp_local_batch_size}},
    //         //     //     {"end_id", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, end_ids_buf_}},
    //         //     //     {"is_initialize_random_table", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &is_initialize_random_table}}};

    //         //     // for (auto t = input_tensors->begin(); t != input_tensors->end(); ++t) {
    //         //     //     if (dynamic_decode_input_tensors.find(t->first) == dynamic_decode_input_tensors.end()) {
    //         //     //         dynamic_decode_input_tensors.insert(*t);
    //         //     //     }
    //         //     // }

    //         //     // // common outputs
    //         //     // bool                                    subbatch_should_stop = false;
    //         //     // std::unordered_map<std::string, Tensor> dynamic_decode_output_tensors{
    //         //     //     {"output_ids",
    //         //     //      Tensor{MEMORY_GPU, TYPE_INT32, {total_seq_len, batch_size, beam_width}, output_ids_buf_}},
    //         //     //     {"finished", Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width}, finished_buf_}},
    //         //     //     // cum_log_probs is necessary for beam search, while it is optional for sampling.
    //         //     //     {"cum_log_probs",
    //         //     //      Tensor{MEMORY_GPU,
    //         //     //             TYPE_FP32,
    //         //     //             {batch_size * beam_width},
    //         //     //             ((beam_width > 1) || (output_tensors->count("cum_log_probs") > 0)) ? cum_log_probs_ :
    //         //     //                                                                                  nullptr}},
    //         //     //     {"output_log_probs",
    //         //     //      Tensor{MEMORY_GPU,
    //         //     //             TYPE_FP32,
    //         //     //             {total_seq_len, batch_size, beam_width},
    //         //     //             output_tensors->count("output_log_probs") > 0
    //         //     //                     && output_tensors->at("output_log_probs").data != nullptr ?
    //         //     //                 output_log_probs_buf_ :
    //         //     //                 nullptr}},
    //         //     //     {"parent_ids",
    //         //     //      Tensor{MEMORY_GPU, TYPE_INT32, {total_seq_len, batch_size, beam_width}, parent_ids_buf_}},
    //         //     //     {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, sequence_lengths_}},
    //         //     //     {"tgt_cache_indirection",
    //         //     //      Tensor{MEMORY_GPU,
    //         //     //             TYPE_INT32,
    //         //     //             {local_batch_size, beam_width, max_output_seq_len},
    //         //     //             cache_indirections_[tgt_indir_idx] + id_offset * max_output_seq_len}},
    //         //     //     {"should_stop", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &subbatch_should_stop}}};

    //         //     // for (auto t = output_tensors->begin(); t != output_tensors->end(); ++t) {
    //         //     //     // Handle exceptions.
    //         //     //     if (t->first == "cum_log_probs" || t->first == "output_log_probs") {
    //         //     //         continue;
    //         //     //     }
    //         //     //     dynamic_decode_output_tensors.insert(*t);
    //         //     // }

    //         //     // dynamic_decode_layer_->forward(&dynamic_decode_output_tensors, &dynamic_decode_input_tensors);
    //         //     // *generation_should_stop_ &= subbatch_should_stop;

    //         // }
        
    //     }

    //     // if (pipeline_para_.world_size_ > 1) {
    //     //     ftNcclGroupStart();
    //     //     ftNcclBroadCast(output_ids_buf_ + step * batch_size * beam_width,
    //     //                     batch_size * beam_width,
    //     //                     pipeline_para_.world_size_ - 1,
    //     //                     pipeline_para_,
    //     //                     stream_);

    //     //     ftNcclBroadCast(
    //     //         sequence_lengths_, batch_size * beam_width, pipeline_para_.world_size_ - 1, pipeline_para_, stream_);

    //     //     ftNcclBroadCast(generation_should_stop_, 1, pipeline_para_.world_size_ - 1, pipeline_para_, stream_);

    //     //     if (beam_width > 1) {
    //     //         ftNcclBroadCast(cache_indirections_[tgt_indir_idx],
    //     //                         batch_size * beam_width * max_output_seq_len,
    //     //                         pipeline_para_.world_size_ - 1,
    //     //                         pipeline_para_,
    //     //                         stream_);
    //     //     }
    //     //     ftNcclGroupEnd();
    //     //     // throw errors when detected
    //     //     ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
    //     //     sync_check_cuda_error();
    //     // }

    //     // if (*generation_should_stop_) {
    //     //     break;
    //     // }
    //     // if (token_generated_cb_ && step + 1 < (int)max_output_seq_len) {
    //     //     setOutputTensors(output_tensors, input_tensors, max_input_length, max_output_seq_len);
    //     //     sendTensorsToFirstPipelineNode(output_tensors, input_tensors);

    //     //     if (pipeline_para_.rank_ == 0 && tensor_para_.rank_ == 0) {
    //     //         token_generated_cb_(output_tensors, token_generated_ctx_);
    //     //     }
    //     // }
    //     // if (step == max_input_length) {
    //     //     /* We have just finished processing input: update the padding count:
    //     //      * total_padding_count += (max_input_length - input_lengths)
    //     //      * if has prefix prompts, += (max_prefix_prompt_length - prompt_length)
    //     //      */
    //     //     invokeUpdatePaddingCount(tiled_total_padding_count_,
    //     //                              input_tensors->at("input_lengths").getPtr<const int>(),  // not_tiled
    //     //                              has_prefix_prompt_ ? tiled_prompt_lengths_buf_ : (const int*)nullptr,
    //     //                              max_input_length,
    //     //                              has_prefix_prompt_ ? max_prefix_prompt_length : 0,
    //     //                              batch_size,
    //     //                              beam_width,
    //     //                              stream_);
    //     // }
    // }
    
    // for(auto& pair : InferenceStatus) {
    //     pair.second.reset();
    // }

}



template<typename T>
void GptJInference<T>::forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                               const std::unordered_map<std::string, Tensor>* input_tensors,
                               const std::vector<GptJDecoderLayerWeight<T>>*  gpt_decoder_layer_weight)
{
    // input tensors:
    //      decoder_input [local_batch_size, hidden_dimension],
    //      finished [local_batch_size],
    //      sequence_lengths [local_batch_size]
    //      total_padding_tokens [local_batch_size],
    //      d_prefix_prompt_lengths [local_batch_size],on GPU
    //      max_prefix_prompt_length [1] on cpu
    //      max_input_length [1] on cpu
    //      step [1] on cpu
    //      ite [1] on cpu
    //      cache_indirection [local_batch_size / beam_width, beam_width, memory_len]
    //              Here, local_batch_size contains the beam_width, so local_batch_size / beam_width
    //              is real local_batch_size.
    //      masked_tokens[local_batch_size, memory_len]

    // output tensors:
    //      decoder_output [local_batch_size, hidden_dimension],
    //      key_cache [num_layer, batch_size, head_num, size_per_head // x, memory_len, x]
    //      value_cache [num_layer, batch_size, head_num, memory_len, size_per_head]

    FT_CHECK(input_tensors->size() == 11);
    FT_CHECK(output_tensors->size() == 3);

    const DataType data_type        = getTensorType<T>();
    const size_t   local_batch_size = input_tensors->at("decoder_input").shape[0];
    const int      ite              = input_tensors->at("ite").getVal<const int>();

    T* decoder_input  = input_tensors->at("decoder_input").getPtr<T>();
    T* decoder_output = output_tensors->at("decoder_output").getPtr<T>();

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

    
    for (uint l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l) == false) {
            continue;
        }
        T* layer_input  = (l == 0) ? decoder_input : decoder_layer_output_;
        T* layer_output = (l == num_layer_ - 1) ? decoder_output : decoder_layer_output_;

        if (isFirstLayerParallelId(l) == true && pipeline_para_.rank_ != 0 && pipeline_para_.world_size_ > 1) {
            int data_size = local_batch_size * hidden_units_ / tensor_para_.world_size_;
            // ftNcclRecv(layer_input, local_batch_size * hidden_units_, pipeline_para_.rank_ - 1, pipeline_para_,
            // stream_);
            ftNcclRecv(layer_input + data_size * tensor_para_.rank_,
                       data_size,
                       pipeline_para_.rank_ - 1,
                       pipeline_para_,
                       stream_);
            if (tensor_para_.world_size_ > 1) {
                ftNcclAllGather(layer_input, layer_input, data_size, tensor_para_.rank_, tensor_para_, stream_);
            }
        }

        invokeGeneralLayerNorm(decoder_normed_input_,
                               layer_input,
                               gpt_decoder_layer_weight->at(l).pre_layernorm_weights.gamma,
                               gpt_decoder_layer_weight->at(l).pre_layernorm_weights.beta,
                               layernorm_eps_,
                               local_batch_size,
                               hidden_units_,
                               (float*)nullptr,
                               0,
                               stream_);
        sync_check_cuda_error();

        TensorMap self_attention_input_tensors(*input_tensors);
        self_attention_input_tensors.insert(
            "input_query", Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, decoder_normed_input_});

        size_t cache_offset = l - getFirstLayerParallelId();
        for (auto t = k_cache.shape.begin() + 1; t != k_cache.shape.end(); ++t) {
            cache_offset *= *t;
        };
        size_t ite_cache_offset = ite * local_batch_size;
        for (auto t = k_cache.shape.begin() + 2; t != k_cache.shape.end(); ++t) {
            ite_cache_offset *= *t;
        }
        cache_offset += ite_cache_offset;
        
        TensorMap self_attention_output_tensors{
            {"hidden_features", Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, self_attn_output_}},
            {"key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_size, k_cache.getPtrWithOffset(cache_offset)}},
            {"value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_size, v_cache.getPtrWithOffset(cache_offset)}}};

        self_attention_layer_->forward(&self_attention_output_tensors,
                                       &self_attention_input_tensors,
                                       &gpt_decoder_layer_weight->at(l).self_attention_weights);

        TensorMap ffn_input_tensors(
            {{"ffn_input", Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, decoder_normed_input_}}});
        TensorMap ffn_output_tensors(
            {{"ffn_output", Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, ffn_output_}}});
        ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &gpt_decoder_layer_weight->at(l).ffn_weights);

        invokeAddBiasAttentionFfnResidual(layer_output,
                                          ffn_output_,
                                          self_attn_output_,
                                          layer_input,
                                          gpt_decoder_layer_weight->at(l).ffn_weights.output_weight.bias,
                                          local_batch_size,
                                          hidden_units_,
                                          stream_);
        sync_check_cuda_error();

        if (isLastLayerParallelId(l) == true && pipeline_para_.rank_ != pipeline_para_.world_size_ - 1
            && pipeline_para_.world_size_ > 1) {
            int data_size = local_batch_size * hidden_units_ / tensor_para_.world_size_;

            ftNcclSend(layer_output + data_size * tensor_para_.rank_,
                       data_size,
                       pipeline_para_.rank_ + 1,
                       pipeline_para_,
                       stream_);
        }
    
        // fprintf(stdout, "I am here\n");
    }

}

template class GptJInference<float>;
template class GptJInference<half>;
#ifdef ENABLE_BF16
template class GptJInference<__nv_bfloat16>;
#endif

}