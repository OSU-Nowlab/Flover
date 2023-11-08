#include "llama_inference.h"
#include "src/flover/layers/TensorParallelSiluFfnLayer.h"
#include "src/flover/layers/attention_layers/TensorParallelDecoderSelfAttentionLayer.h"
#include "src/flover/kernels/bert_preprocess_kernels.h"
#include "src/flover/kernels/decoding_kernels.h"
#include "src/flover/kernels/gpt_kernels.h"
#include "src/flover/layers/beam_search_layers/BaseBeamSearchLayer.h"
#include "src/flover/utils/memory_shuffler.h"
#include <algorithm>
#include <fstream>
#include <iomanip>

class ProgressBar {
private:
    int total_iterations;
    int bar_width;
    int max_actives;
    int progress_bar_thickness;
    std::vector<int> active_bars;
    int log_line1, log_line2; // keep track of where to print next log

public:
    ProgressBar(int total, int width = 50, int maxActives = 32, int thickness = 3) : 
        total_iterations(total*1.5), bar_width(width*2), max_actives(maxActives), progress_bar_thickness(thickness), active_bars(width*2, 0), log_line1(6), log_line2(6) {
        // Initial screen clear
        std::cout << "\033[2J\033[1;1H";  // ANSI escape codes to clear screen and move to top-left corner
    }

    void update(int i, int actives, const std::string& logMessage = "", int req_id = 0) {
        int pos = (float)i / total_iterations * bar_width;
        
        // Position cursor to the beginning
        std::cout << "\033[1;1H";

        // Progress bar for total iters
        for (int k = 0; k < progress_bar_thickness - 1; ++k) {
            std::cout << "[";
            for (int j = 0; j < bar_width; ++j) {
                if (j < pos) std::cout << "=";
                else if (j == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "\n";
        }
        std::cout << "[";
        for (int j = 0; j < bar_width; ++j) {
            if (j < pos) std::cout << "=";
            else if (j == pos) std::cout << ">";
        }
        std::cout << " current iters=" << i << "\n";

        // Record the actives value for this position
        active_bars[pos] = actives;

        // Actives title
        std::cout << "\n# of In-flight requests " << "                                                                                               Incoming requests" << "                      Finished requests\n";

        // Draw all active bars with y-axis labels
        for (int height = max_actives; height >= 0; --height) {  
            std::cout << std::setw(3) << height << "|";  // 3 spaces wide for y-axis labels
            for (int j = 0; j <= pos; ++j) {
                if (active_bars[j] >= height) std::cout << "█";
                else std::cout << " ";
            }
            std::cout << "\n";
        }

        // Draw x-axis
        std::cout << "    +";
        for (int j = 0; j < bar_width; ++j) {
            std::cout << "-";
        }
        std::cout << "> iters\n";

        if (logMessage == "request added...") {
            std::cout << "\033[" << log_line1 << ";120H" << "request " << req_id << " added...";  // Position cursor to log_line, column 60
            log_line1++;
        } else if(logMessage == "request finished...") {
            std::cout << "\033[" << log_line2 << ";159H" << "request " << req_id << " finished...";  // Position cursor to log_line, column 60
            log_line2++;
        }

        std::cout.flush();
    }

    void finish() {
        std::cout << "\nFinished!\n";
    }
};

namespace flover {

template<typename T>
void LlamaInference<T>::initialize()
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
                                                                           !use_gptj_residual_,
                                                                           is_free_buffer_after_forward_,
                                                                           false,
                                                                           int8_mode_,
                                                                           custom_all_reduce_comm_,
                                                                           enable_custom_all_reduce_);

    ffn_layer_ = new TensorParallelSiluFfnLayer<T>(max_batch_size_,
                                                   1,
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

    dynamic_decode_layer_ = new DynamicDecodeLayer<float>(vocab_size_,
                                                          vocab_size_padded_,
                                                          0,  // end_id, deprecated
                                                          stream_,
                                                          cublas_wrapper_,
                                                          nullptr,
                                                          is_free_buffer_after_forward_,
                                                          cuda_device_prop_);
}

template<typename T>
void LlamaInference<T>::allocateBuffer()
{
}

template<typename T>
void LlamaInference<T>::freeBuffer()
{
}


template<typename T>
bool LlamaInference<T>::isValidLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T>
bool LlamaInference<T>::isFirstLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T>
bool LlamaInference<T>::isLastLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T>
int LlamaInference<T>::getFirstLayerParallelId()
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T>
LlamaInference<T>::LlamaInference(size_t                              max_batch_size,
                                    size_t                              head_num,
                                    size_t                              size_per_head,
                                    size_t                              inter_size,
                                    size_t                              num_layer,
                                    int                                 vocab_size,
                                    size_t                              rotary_embedding_dim,
                                    bool                                neox_rotary_style,
                                    bool                                use_gptj_residual,
                                    float                               layernorm_eps,
                                    NcclParam                           tensor_para,
                                    NcclParam                           pipeline_para,
                                    cudaStream_t                        stream,
                                    cublasMMWrapper*                    cublas_wrapper,
                                    bool                                is_free_buffer_after_forward,
                                    int                                 int8_mode,
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
    use_gptj_residual_(use_gptj_residual),
    layernorm_eps_(layernorm_eps),
    hidden_units_(head_num_ * size_per_head),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    int8_mode_(int8_mode),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    int local_vacab_size = ceil(vocab_size_ / 1.f / tensor_para_.world_size_);
    if (std::is_same<half, T>::value
#ifdef ENABLE_BF16
        || std::is_same<__nv_bfloat16, T>::value
#endif
    ) {
        local_vacab_size = ceil(local_vacab_size / 8.f) * 8;
    }
    vocab_size_padded_ = (size_t)local_vacab_size * tensor_para_.world_size_;
    initialize();
}

template<typename T>
LlamaInference<T>::~LlamaInference()
{
    delete self_attention_layer_;
    delete ffn_layer_;
}


template<typename T>
void LlamaInference<T>::run(int* stop_flag, 
                           const INIReader& reader, 
                           const LlamaWeight<T>* gpt_weights, 
                           InferenceQueue* inque, 
                           bool* new_req, 
                           bool* added, 
                           int use_mem_shuffle)
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
    int               step                     = max_cache_seq_len - 1; // modify this for dynamic decoder, each request uses its own step
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

    int max_output_seq_len = total_seq_len;
    int max_input_length   = fixed_input_len;
    
    const DataType       data_type      = getTensorType<T>();
    const cudaDataType_t gemm_data_type = getCudaDataType<T>();

    std::string file_path = "/home/yao.877/parallel_inference/projects/FasterTransformer_AR/Flover/flover_runtime/request_elapsed/" + model_name + "_" + std::to_string(max_concurrency);
    std::ofstream file(file_path);
    
    int total_workload = max_concurrency * per_batch_size * (total_seq_len - fixed_input_len);
    int cum_workload = 0;
    const int interval = reader.GetInteger("runtime_hyperparameter", "interval");
    std::string progree_file_path = "/home/yao.877/parallel_inference/projects/FasterTransformer_AR/Flover/flover_runtime/_" + std::to_string(interval) + "_progress.csv";
    std::ofstream progree_file(progree_file_path);
    std::chrono::high_resolution_clock::time_point start;
    bool first_start = true;

    int temp;
    total_iters = fixed_input_len;

    /* memory trace */
    std::vector<int> mem_status_pre;
    std::vector<int> mem_status;
    for(int i=0;i<max_concurrency;++i){
        mem_status_pre.push_back(-1);
        mem_status.push_back(-1);
    }
    
    int cur_low_id = 0;
    int cur_high_id = -1;
    int high_id_offset = 0;
    int total_mem_move_amount = 0;

    ProgressBar pb(1230);
    bool added_req = false;
    int added_id = 0;
    bool removed_req = false;
    int removed_id = 0;

    while (true) {
        InferenceInfo _req;
        while (*new_req) {
            if (*added) {
                *new_req = false;
                *added = false;
                break;
            }
        }
        if (inque->get(_req)) {
            // log("Rank ", tensor_para_.rank_, " :: -- received task id ", _req.unique_task_id);
            if (first_start) {
                start = std::chrono::high_resolution_clock::now();
                first_start = false;
                }
            added_req = true;
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

                normed_decoder_output_buf_len += _req.normed_decoder_output_buf_len;
                logits_buf_len                += _req.logits_buf_len;
                nccl_logits_buf_len           += _req.nccl_logits_buf_len;

                decoder_layer_output_len      += _req.decoder_layer_output_len;
                decoder_normed_input_len      += _req.decoder_normed_input_len;
                self_attn_output_len          += _req.self_attn_output_len;
                ffn_output_len                += _req.ffn_output_len;

                _req.mem_id                   -= high_id_offset;
                
                
                InferenceInfo* heap_req       = new InferenceInfo(_req);
                std::unique_ptr<InferenceInfo> ptr(heap_req);
                InferenceStatus[_req.unique_task_id] = std::move(ptr);

                mem_status_pre[_req.mem_id]   = _req.unique_task_id;
                mem_status[_req.mem_id]       = _req.unique_task_id;
                added_id = _req.unique_task_id;
                if(_req.mem_id > cur_high_id) 
                    cur_high_id = _req.mem_id;

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
                // log("Rank ", tensor_para_.rank_, " Inference request ", _req.unique_task_id, " prepare done ");
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

            invokeGeneralT5LayerNorm(normed_decoder_output_buf_ + normed_decoder_output_buf_offset,
                                        decoder_output_buf_ + decoder_output_buf_offset,
                                        gpt_weights->post_decoder_layernorm.gamma,
                                        (const T*)nullptr,
                                        layernorm_eps_,
                                        beam_width,
                                        hidden_units_,
                                        stream_);
            sync_check_cuda_error();

            if (tensor_para_.world_size_ == 1) {
                float alpha = 1.0f;
                float beta  = 0.0f;
                cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        vocab_size_padded_,  // n
                                        running_batchsize * beam_width,
                                        hidden_units_,  // k
                                        &alpha,
                                        padded_embedding_kernel_ptr_,
                                        gemm_data_type,
                                        hidden_units_,  // k
                                        normed_decoder_output_buf_ + normed_decoder_output_buf_offset,
                                        gemm_data_type,
                                        hidden_units_,  // k
                                        &beta,
                                        logits_buf_ + logits_buf_offset,
                                        CUDA_R_32F,
                                        vocab_size_padded_, /* n */
                                        CUDA_R_32F,
                                        cublasGemmAlgo_t(-1));
            }
            else {
                FT_CHECK(vocab_size_padded_ % tensor_para_.world_size_ == 0);
                const int local_vocab_size = vocab_size_padded_ / tensor_para_.world_size_;
                float     alpha            = 1.0f;
                float     beta             = 0.0f;
                cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        local_vocab_size,  // n
                                        running_batchsize * beam_width,
                                        hidden_units_,  // k
                                        &alpha,
                                        padded_embedding_kernel_ptr_
                                            + tensor_para_.rank_ * local_vocab_size * hidden_units_,
                                        gemm_data_type,
                                        hidden_units_,  // k
                                        normed_decoder_output_buf_ + normed_decoder_output_buf_offset,
                                        gemm_data_type,
                                        hidden_units_,  // k
                                        &beta,
                                        nccl_logits_buf_ + nccl_logits_buf_offset + tensor_para_.rank_ * running_batchsize * beam_width * local_vocab_size,
                                        CUDA_R_32F,
                                        local_vocab_size, /* n */
                                        CUDA_R_32F,
                                        cublasGemmAlgo_t(-1));
                

                ftNcclAllGather(nccl_logits_buf_ + nccl_logits_buf_offset,
                                nccl_logits_buf_ + nccl_logits_buf_offset,
                                running_batchsize * beam_width * local_vocab_size,
                                tensor_para_.rank_,
                                tensor_para_,
                                stream_);
                invokeTransposeAxis01(logits_buf_ + logits_buf_offset,
                                        nccl_logits_buf_ + nccl_logits_buf_offset,
                                        tensor_para_.world_size_,
                                        running_batchsize * beam_width,
                                        local_vocab_size,
                                        stream_);
            }

            for (auto& pair : InferenceStatus) {
                int key = pair.first;
                InferenceInfo* value                      = pair.second.get();
                int            tmp_batch_size             = per_batch_size;
                bool           is_initialize_random_table = step == max_input_length; // TODO: record each request's step
                int            ite = 0;

                std::unordered_map<std::string, Tensor> dynamic_decode_input_tensors{
                    {"logits",
                     Tensor{MEMORY_GPU, TYPE_FP32, {per_batch_size, beam_width, vocab_size_padded_}, logits_buf_ + value->mem_id * value->logits_buf_len}},
                    // {"embedding_bias", Tensor{MEMORY_GPU, data_type, {vocab_size_padded_}, nullptr}},
                    {"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step}},
                    {"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length}},
                    {"input_lengths",
                     Tensor{MEMORY_GPU, TYPE_INT32, {per_batch_size, beam_width}, tiled_input_lengths_buf_ + value->mem_id * value->tiled_input_lengths_buf_len}},
                    {"sequence_limit_length", Tensor{MEMORY_GPU, TYPE_UINT32, {per_batch_size}, seq_limit_len_ + value->mem_id * value->seq_limit_len_len}},
                    {"ite", Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &ite}},
                    {"src_cache_indirection",
                     Tensor{MEMORY_GPU,
                            TYPE_INT32,
                            {per_batch_size, beam_width, max_output_seq_len},
                            cache_indirections_[src_indir_idx] + value->mem_id * value->cache_indirections_len}},
                    {"local_batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &tmp_batch_size}},
                    {"end_id", Tensor{MEMORY_GPU, TYPE_INT32, {per_batch_size}, end_ids_buf_ + value->mem_id * value->end_ids_buf_len}},
                    {"is_initialize_random_table", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &is_initialize_random_table}}};

                for (auto t = value->input_tensors->begin(); t != value->input_tensors->end(); ++t) {
                    if (dynamic_decode_input_tensors.find(t->first) == dynamic_decode_input_tensors.end()) {
                        dynamic_decode_input_tensors.insert(*t);
                    }
                }
                
                // common outputs
                bool                                    subbatch_should_stop = false;
                std::unordered_map<std::string, Tensor> dynamic_decode_output_tensors{
                    {"output_ids",
                     Tensor{MEMORY_GPU, TYPE_INT32, {total_seq_len, per_batch_size, beam_width}, output_ids_buf_ + value->mem_id * value->output_ids_buf_len}},
                    {"finished", Tensor{MEMORY_GPU, TYPE_BOOL, {per_batch_size * beam_width}, finished_buf_ + value->mem_id * value->finished_buf_len}},
                    // cum_log_probs is necessary for beam search, while it is optional for sampling.
                    {"cum_log_probs",
                     Tensor{MEMORY_GPU,
                            TYPE_FP32,
                            {per_batch_size * beam_width},
                            ((beam_width > 1) || (value->output_tensors->count("cum_log_probs") > 0)) ? cum_log_probs_ + value->mem_id * value->cum_log_probs_len :
                                                                                                 nullptr}},
                    {"output_log_probs",
                     Tensor{MEMORY_GPU,
                            TYPE_FP32,
                            {total_seq_len, per_batch_size, beam_width},
                            value->output_tensors->count("output_log_probs") > 0
                                    && value->output_tensors->at("output_log_probs").data != nullptr ?
                                output_log_probs_buf_ + value->mem_id * value->output_log_probs_buf_len :
                                nullptr}},
                    {"parent_ids",
                     Tensor{MEMORY_GPU, TYPE_INT32, {total_seq_len, per_batch_size, beam_width}, parent_ids_buf_ + value->mem_id * value->parent_ids_buf_len}},
                    {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, {per_batch_size * beam_width}, sequence_lengths_ + value->mem_id * value->sequence_lengths_len}},
                    {"tgt_cache_indirection",
                     Tensor{MEMORY_GPU,
                            TYPE_INT32,
                            {per_batch_size, beam_width, max_output_seq_len},
                            cache_indirections_[tgt_indir_idx] + value->mem_id * value->cache_indirections_len}},
                    {"should_stop", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &subbatch_should_stop}}};

                for (auto t = value->output_tensors->begin(); t != value->output_tensors->end(); ++t) {
                    // Handle exceptions.
                    if (t->first == "cum_log_probs" || t->first == "output_log_probs") {
                        continue;
                    }
                    dynamic_decode_output_tensors.insert(*t);
                }
                if (!dynamic_decoder_set) { 
                    TensorMap _input_map(*(value->input_tensors));
                    dynamic_decode_layer_->setup(per_batch_size, beam_width, &_input_map);
                    dynamic_decoder_set = true;
                    cudaStreamSynchronize(stream_);
                }
                dynamic_decode_layer_->forward(&dynamic_decode_output_tensors, &dynamic_decode_input_tensors);
                    
            }

            total_iters += 1;
            // log("Rank ", tensor_para_.rank_, " Total iters ", total_iters, ", actives ", actives, ", working batch size ", running_batchsize * beam_width, ", cache ind ", src_indir_idx);
            if (added_req) {
                pb.update(total_iters, actives, "request added...", added_id);
                added_req = false;
            } else if (removed_req) {
                pb.update(total_iters, actives, "request finished...", removed_id);
                removed_req = false;
            }
            else {
                pb.update(total_iters, actives, "");
            }


            int this_time_count_move = 0;
            std::vector<int> remove_request_id;
            for (auto it = InferenceStatus.begin(); it != InferenceStatus.end(); /* no increment here */) {
                int key              = it->first;
                InferenceInfo* value = it->second.get();
                value->cur_step      += 1;


                if(value->cur_step == value->defined_generation_len || value->cur_step >= value->total_seq_len){
                    removed_req = true;
                    removed_id = key;
                    actives                          -=1;
                    // log("Request ", key, " reaches len ", value->cur_step);
                    if (value->mem_id == cur_low_id || value->mem_id == cur_high_id || use_mem_shuffle) {
                        running_batchsize                -= per_batch_size;
                        self_k_cache_shape.at(1)         = running_batchsize * beam_width;
                        self_v_cache_shape.at(1)         = running_batchsize * beam_width;
                        decoder_input_buf_len            -= value->decoder_input_buf_len;
                        output_ids_buf_len               -= value->output_ids_buf_len;
                        tiled_total_padding_count_len    -= value->tiled_total_padding_count_len;
                        finished_buf_len                 -= value->finished_buf_len;
                        sequence_lengths_len             -= value->sequence_lengths_len;
                        tiled_prompt_lengths_buf_len     -= value->tiled_prompt_lengths_buf_len;
                        cache_indirections_len           -= value->cache_indirections_len;
                        masked_tokens_len                -= value->masked_tokens_len;
                        decoder_output_buf_len           -= value->decoder_output_buf_len;
                        key_cache_len                    -= value->key_cache_len;
                        value_cache_len                  -= value->value_cache_len;
                        normed_decoder_output_buf_len    -= value->normed_decoder_output_buf_len;
                        logits_buf_len                   -= value->logits_buf_len;
                        nccl_logits_buf_len              -= value->nccl_logits_buf_len;
                        decoder_layer_output_len         -= value->decoder_layer_output_len;
                        decoder_normed_input_len         -= value->decoder_normed_input_len;
                        self_attn_output_len             -= value->self_attn_output_len;
                        ffn_output_len                   -= value->ffn_output_len;

                        if (value->mem_id == cur_low_id) {
                            decoder_input_buf_offset         += value->decoder_input_buf_len;
                            output_ids_buf_offset            += value->output_ids_buf_len;
                            tiled_total_padding_count_offset += value->tiled_total_padding_count_len;
                            finished_buf_offset              += value->finished_buf_len;
                            sequence_lengths_offset          += value->sequence_lengths_len;
                            tiled_prompt_lengths_buf_offset  += value->tiled_prompt_lengths_buf_len;
                            cache_indirections_offset        += value->cache_indirections_len;
                            masked_tokens_offset             += value->masked_tokens_len;
                            decoder_output_buf_offset        += value->decoder_output_buf_len;
                            key_cache_offset                 += value->key_cache_len;
                            value_cache_offset               += value->value_cache_len;
                            normed_decoder_output_buf_offset += value->normed_decoder_output_buf_len;
                            logits_buf_offset                += value->logits_buf_len;
                            nccl_logits_buf_offset           += value->nccl_logits_buf_len;
                            decoder_layer_output_offset      += value->decoder_layer_output_len;
                            decoder_normed_input_offset      += value->decoder_normed_input_len;
                            self_attn_output_offset          += value->self_attn_output_len;
                            ffn_output_offset                += value->ffn_output_len;
                            cur_low_id++;
                        } else if (value->mem_id == cur_high_id) {
                            high_id_offset++;
                            cur_high_id--;
                        } else {
                            mem_status_pre[value->mem_id] = -1;
                            this_time_count_move++;
                        }
                    }
                    remove_request_id.push_back(key);
                    if (tensor_para_.rank_ == 0) {
                        long long finish = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                        long long elapsed = finish - value->start_time;
                        file << std::to_string(elapsed) << std::endl;
                    }
                    it++;
                    // log("Rank ", tensor_para_.rank_, " task ", key, " finished");    
                    // break;
                }else {
                    ++it;
                }
            }

            if (use_mem_shuffle && this_time_count_move) {
                std::map<int, int> mem_move_map;
                shuffler(mem_status_pre, mem_move_map);
                if(!mem_move_map.size()){
                    for(int cp_m=0;cp_m<mem_status.size();++cp_m){
                        mem_status[cp_m] = mem_status_pre[cp_m];
                    }
                }
                for (const auto &pair : mem_move_map) {
                    total_mem_move_amount++;
                    InferenceStatus.at(mem_status[pair.first])->mem_id = pair.second;
                    mem_status[pair.second] = mem_status[pair.first];
                    mem_status_pre[pair.second] = mem_status[pair.first];
                    mem_status[pair.first] = -1;
                    mem_status_pre[pair.first] = -1;

                    cudaMemcpyAsync(decoder_input_buf_ + pair.second * _req.decoder_input_buf_len, decoder_input_buf_ + pair.first * _req.decoder_input_buf_len, sizeof(T) * _req.decoder_input_buf_len, cudaMemcpyDeviceToDevice, stream_);
                    
                    cudaMemcpyAsync(output_ids_buf_ + pair.second * _req.output_ids_buf_len, output_ids_buf_ + pair.first * _req.output_ids_buf_len, sizeof(int) * _req.output_ids_buf_len, cudaMemcpyDeviceToDevice, stream_);
                    
                    cudaMemcpyAsync(tiled_total_padding_count_ + pair.second * _req.tiled_total_padding_count_len, tiled_total_padding_count_ + pair.first * _req.tiled_total_padding_count_len, sizeof(int) * _req.tiled_total_padding_count_len, cudaMemcpyDeviceToDevice, stream_);
                    
                    cudaMemcpyAsync(finished_buf_ + pair.second * _req.finished_buf_len, finished_buf_ + pair.first * _req.finished_buf_len, sizeof(bool) * _req.finished_buf_len, cudaMemcpyDeviceToDevice, stream_);

                    cudaMemcpyAsync(sequence_lengths_ + pair.second * _req.sequence_lengths_len, sequence_lengths_ + pair.first * _req.sequence_lengths_len, sizeof(int) * _req.sequence_lengths_len, cudaMemcpyDeviceToDevice, stream_);

                    cudaMemcpyAsync(tiled_prompt_lengths_buf_ + pair.second * _req.tiled_prompt_lengths_buf_len, tiled_prompt_lengths_buf_ + pair.first * _req.tiled_prompt_lengths_buf_len, sizeof(int) * _req.tiled_prompt_lengths_buf_len, cudaMemcpyDeviceToDevice, stream_);
                    
                    cudaMemcpyAsync(cache_indirections_ + pair.second * _req.cache_indirections_len, cache_indirections_ + pair.first * _req.cache_indirections_len, sizeof(int) * _req.cache_indirections_len, cudaMemcpyDeviceToDevice, stream_);
                    
                    cudaMemcpyAsync(masked_tokens_ + pair.second * _req.masked_tokens_len, masked_tokens_ + pair.first * _req.masked_tokens_len, sizeof(bool) * _req.masked_tokens_len, cudaMemcpyDeviceToDevice, stream_);
                    
                    cudaMemcpyAsync(decoder_output_buf_ + pair.second * _req.decoder_output_buf_len, decoder_output_buf_ + pair.first * _req.decoder_output_buf_len, sizeof(T) * _req.decoder_output_buf_len, cudaMemcpyDeviceToDevice, stream_);
                    
                    cudaMemcpyAsync(key_cache_ + pair.second * _req.key_cache_len, key_cache_ + pair.first * _req.key_cache_len, sizeof(T) * _req.key_cache_len, cudaMemcpyDeviceToDevice, stream_);
                    
                    cudaMemcpyAsync(value_cache_ + pair.second * _req.value_cache_len, value_cache_ + pair.first * _req.value_cache_len, sizeof(T) * _req.value_cache_len, cudaMemcpyDeviceToDevice, stream_);
                    
                    cudaMemcpyAsync(normed_decoder_output_buf_ + pair.second * _req.normed_decoder_output_buf_len, normed_decoder_output_buf_ + pair.first * _req.normed_decoder_output_buf_len, sizeof(T) * _req.normed_decoder_output_buf_len, cudaMemcpyDeviceToDevice, stream_);
                    
                    cudaMemcpyAsync(logits_buf_ + pair.second * _req.logits_buf_len, logits_buf_ + pair.first * _req.logits_buf_len, sizeof(float) * _req.logits_buf_len, cudaMemcpyDeviceToDevice, stream_);
                    
                    cudaMemcpyAsync(nccl_logits_buf_ + pair.second * _req.nccl_logits_buf_len, nccl_logits_buf_ + pair.first * _req.nccl_logits_buf_len, sizeof(float) * _req.nccl_logits_buf_len, cudaMemcpyDeviceToDevice, stream_);
                    
                    cudaMemcpyAsync(decoder_layer_output_ + pair.second * _req.decoder_layer_output_len, decoder_layer_output_ + pair.first * _req.decoder_layer_output_len, sizeof(T) * _req.decoder_layer_output_len, cudaMemcpyDeviceToDevice, stream_);
                    
                    cudaMemcpyAsync(decoder_normed_input_ + pair.second * _req.decoder_normed_input_len, decoder_normed_input_ + pair.first * _req.decoder_normed_input_len, sizeof(T) * _req.decoder_normed_input_len, cudaMemcpyDeviceToDevice, stream_);
                    
                    cudaMemcpyAsync(self_attn_output_ + pair.second * _req.self_attn_output_len, self_attn_output_ + pair.first * _req.self_attn_output_len, sizeof(T) * _req.self_attn_output_len, cudaMemcpyDeviceToDevice, stream_);
                    
                    cudaMemcpyAsync(ffn_output_ + pair.second * _req.ffn_output_len, ffn_output_ + pair.first * _req.ffn_output_len, sizeof(T) * _req.ffn_output_len, cudaMemcpyDeviceToDevice, stream_);
                }
                int start_mem_id = 0;
                for(;start_mem_id<mem_status.size();++start_mem_id){
                    if(mem_status[start_mem_id] >= 0) break;
                }
                decoder_input_buf_offset         = start_mem_id * _req.decoder_input_buf_len;
                output_ids_buf_offset            = start_mem_id * _req.output_ids_buf_len;
                tiled_total_padding_count_offset = start_mem_id * _req.tiled_total_padding_count_len;
                finished_buf_offset              = start_mem_id * _req.finished_buf_len;
                sequence_lengths_offset          = start_mem_id * _req.sequence_lengths_len;
                tiled_prompt_lengths_buf_offset  = start_mem_id * _req.tiled_prompt_lengths_buf_len;
                cache_indirections_offset        = start_mem_id * _req.cache_indirections_len;
                masked_tokens_offset             = start_mem_id * _req.masked_tokens_len;
                decoder_output_buf_offset        = start_mem_id * _req.decoder_output_buf_len;
                key_cache_offset                 = start_mem_id * _req.key_cache_len;
                value_cache_offset               = start_mem_id * _req.value_cache_len;
                normed_decoder_output_buf_offset = start_mem_id * _req.normed_decoder_output_buf_len;
                logits_buf_offset                = start_mem_id * _req.logits_buf_len;
                nccl_logits_buf_offset           = start_mem_id * _req.nccl_logits_buf_len;
                decoder_layer_output_offset      = start_mem_id * _req.decoder_layer_output_len;
                decoder_normed_input_offset      = start_mem_id * _req.decoder_normed_input_len;
                self_attn_output_offset          = start_mem_id * _req.self_attn_output_len;
                ffn_output_offset                = start_mem_id * _req.ffn_output_len;

            }
            
            for (int ridx=0;ridx<remove_request_id.size();++ridx) {
                InferenceStatus.erase(remove_request_id[ridx]);
            }
        } else {
            if (*stop_flag == 1) {
                InferenceStatus.clear();
                break;
            }
        }
        ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);

        cum_workload += running_batchsize * beam_width;
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        int time = duration.count();
        progree_file << time << "," << static_cast<float>(cum_workload) / total_workload << "\n";

    }
    if (tensor_para_.rank_ == 0 && use_mem_shuffle) {
        // log("Total mem move: ", total_mem_move_amount);
    }
    pb.update(total_iters, actives, "request finished...", removed_id);
    std::cout<<std::endl;
    std::cout.flush();
    std::cout<<std::endl;
    std::cout.flush();
    std::cout<<std::endl;
    std::cout.flush();
}


template<typename T>
void LlamaInference<T>::forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                               const std::unordered_map<std::string, Tensor>* input_tensors,
                               const std::vector<LlamaDecoderLayerWeight<T>*>*  gpt_decoder_layer_weight)
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
        T* layer_input  = (l == 0) ? decoder_input : decoder_layer_output_ + decoder_layer_output_offset;
        T* layer_output = (l == num_layer_ - 1) ? decoder_output : decoder_layer_output_ + decoder_layer_output_offset;

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

        // TODO 使用的是T5 LN，这里是没有int8的参数支持
        invokeGeneralT5LayerNorm(decoder_normed_input_ + decoder_normed_input_offset,
                                 layer_input,
                                 gpt_decoder_layer_weight->at(l)->pre_layernorm_weights.gamma,
                                 (const T*)nullptr,
                                 layernorm_eps_,
                                 local_batch_size,
                                 hidden_units_,
                                 stream_);
        sync_check_cuda_error();

        TensorMap self_attention_input_tensors(*input_tensors);
        self_attention_input_tensors.insert(
            "input_query", Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, decoder_normed_input_ + decoder_normed_input_offset});
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
            {"hidden_features", Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, self_attn_output_ + self_attn_output_offset}},
            {"key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_size, k_cache.getPtrWithOffset(cache_offset)}},
            {"value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_size, v_cache.getPtrWithOffset(cache_offset)}}};

        self_attention_layer_->forward(&self_attention_output_tensors,
                                       &self_attention_input_tensors,
                                       &gpt_decoder_layer_weight->at(l)->self_attention_weights);

        if (use_gptj_residual_) {
            invokeGeneralLayerNorm(decoder_normed_input_ + decoder_normed_input_offset,
                                   layer_input,
                                   gpt_decoder_layer_weight->at(l)->post_attention_layernorm_weights.gamma,
                                   gpt_decoder_layer_weight->at(l)->post_attention_layernorm_weights.beta,
                                   layernorm_eps_,
                                   local_batch_size,
                                   hidden_units_,
                                   (float*)nullptr,
                                   int8_mode_,
                                   stream_);
        }
        else {
            invokeGeneralAddResidualT5PreLayerNorm(
                self_attn_output_ + self_attn_output_offset,
                decoder_normed_input_ + decoder_normed_input_offset,
                layer_input,
                gpt_decoder_layer_weight->at(l)->post_attention_layernorm_weights.gamma,
                layernorm_eps_,
                local_batch_size,
                hidden_units_,
                stream_);
        }

        TensorMap ffn_input_tensors(
            {{"ffn_input", Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, decoder_normed_input_ + decoder_normed_input_offset}}});
        TensorMap ffn_output_tensors(
            {{"ffn_output", Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, ffn_output_ + ffn_output_offset}}});
        ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &gpt_decoder_layer_weight->at(l)->ffn_weights);

        if (use_gptj_residual_) {
            // Original workflow:
            //      layer_output = layer_input + reduceSum(ffn_output + self_attn_output + ffn_output_bias)
            // Our workflow:
            //      layer_output = reduceSum(ffn_output + self_attn_output + ffn_output_bias + layer_input / TP_size)
            // They are equivalent on math, but we can use same buffer for layer_input and layer_output
            invokeAddBiasAttentionFfnResidual(layer_output,
                                              ffn_output_ + ffn_output_offset,
                                              self_attn_output_ + self_attn_output_offset,
                                              layer_input,
                                              gpt_decoder_layer_weight->at(l)->ffn_weights.output_weight.bias,
                                              local_batch_size,
                                              hidden_units_,
                                              tensor_para_.world_size_,
                                              stream_);
            if (tensor_para_.world_size_ > 1) {
                ftNcclAllReduceSum(layer_output, layer_output, local_batch_size * hidden_units_, tensor_para_, stream_);
            }
        }
        else {
            invokeAddBiasResidual(layer_output,
                                  self_attn_output_ + self_attn_output_offset,
                                  gpt_decoder_layer_weight->at(l)->ffn_weights.output_weight.bias,
                                  local_batch_size,
                                  hidden_units_,
                                  stream_);
        }
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
    }

}


template class LlamaInference<float>;
template class LlamaInference<half>;
#ifdef ENABLE_BF16
template class LlamaInference<__nv_bfloat16>;
#endif


}