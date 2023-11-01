#pragma once
#include "3rdparty/INIReader.h"
#include "src/flover/utils/prompt_learning.h"
#include "src/flover/utils/word_list.h"
#include <tbb/concurrent_queue.h>
#include <tbb/tbb.h>
#include <mutex>
#include <random>
#include <cstdlib>
#include <ctime>
#include <map>
#include <string>
#include <vector>

namespace flover {


int fl_read_start_ids(size_t            batch_size,
                   std::vector<int>* v_start_lengths,
                   std::vector<int>* v_start_ids,
                   size_t&           max_input_len,
                   const int         end_id,
                   const int         beam_width,
                   std::string       file_name);

struct RequestInfo {
    size_t             total_seq_len;
    size_t             beam_width;
    uint               top_k;
    float              top_p;
    float              temperature;
    float              repetition_penalty;
    float              presence_penalty;
    float              len_penalty;
    int                min_length;
    size_t             request_batch_size;
    uint32_t           memory_len;
    int                prompt_learning_start_id;
    PromptLearningType prompt_learning_type;
    int                defined_generation_len;
    int                num_tasks;
    float              beam_search_diversity_rate;

    size_t             max_input_len;
    int                start_id;
    int                end_id;
    int                fixed_prompt_len;

    unsigned long long random_seed;
    int                unique_task_id;

    std::vector<int> v_start_lengths;
    std::vector<int> v_start_ids;
    std::vector<int> stop_words;
    std::vector<int> bad_words;
    std::vector<int> tiled_stop_words;


    size_t      head_num;
    size_t      size_per_head;
    size_t      vocab_size;
    size_t      decoder_layers;
    size_t      rotary_embedding_dim;
    size_t      inter_size;
    size_t      hidden_units;
    size_t      tensor_para_size;
    size_t      pipeline_para_size;

    int         generation_len;
    long long start_time;
};

struct RequestQueue {
    tbb::concurrent_queue<RequestInfo> _queue;
    
    void add(RequestInfo value) {
        _queue.push(value);
    }

    void add_dummy(const INIReader& reader, const int task_id, int defined_generation_len, const int random_seed = 0) {

        const std::string model_name           = reader.Get("model_specification", "model_type");
        const size_t      max_concurrency      = reader.GetInteger("model_specification", "max_concurrency");
        const size_t      per_batch_size       = reader.GetInteger("model_specification", "per_batch_size");
        const size_t      total_seq_len          = reader.GetInteger("model_specification", "max_seq_len");
        const size_t      beam_width           = reader.GetInteger("model_specification", "beam_width");
        int               tensor_para_size     = reader.GetInteger("model_specification", "tensor_para_size");
        int               pipeline_para_size     = reader.GetInteger("model_specification", "pipeline_para_size");
        const size_t      max_batch_size       = max_concurrency * per_batch_size;

        const size_t      head_num             = reader.GetInteger(model_name, "head_num");
        const size_t      size_per_head        = reader.GetInteger(model_name, "size_per_head");
        const size_t      vocab_size           = reader.GetInteger(model_name, "vocab_size");
        const size_t      decoder_layers       = reader.GetInteger(model_name, "decoder_layers");
        const size_t      rotary_embedding_dim = reader.GetInteger(model_name, "rotary_embedding");
        const size_t      inter_size           = reader.GetInteger(model_name, "inter_size");
        const size_t      hidden_units         = head_num * size_per_head;

        const uint        top_k                = (uint)reader.GetInteger("runtime_hyperparameter", "top_k");
        const float       top_p                = reader.GetFloat("runtime_hyperparameter", "top_p");
        const float       temperature          = reader.GetFloat("runtime_hyperparameter", "temperature");
        const float       repetition_penalty   = reader.GetFloat("runtime_hyperparameter", "repetition_penalty", 1.0f);
        const float       presence_penalty     = reader.GetFloat("runtime_hyperparameter", "presence_penalty", 0.0f);
        const float       len_penalty          = reader.GetFloat("runtime_hyperparameter", "len_penalty");
        const int         min_length           = reader.GetInteger("runtime_hyperparameter", "min_length", 0);
        const int         start_id             = reader.GetInteger(model_name, "start_id");
        const int         end_id               = reader.GetInteger(model_name, "end_id");
        const float       beam_search_diversity_rate = reader.GetFloat("runtime_hyperparameter", "beam_search_diversity_rate");
        const int         request_output_len   = reader.GetInteger("request", "request_output_len");
        const uint32_t    memory_len           = reader.GetInteger("request", "memory_len", 0);
        const int         num_tasks            = reader.GetInteger(model_name, "num_tasks", 0);
        const size_t      fixed_prompt_len     = reader.GetInteger("model_specification", "fixed_prompt_len");
        
        // Handle bad_words dictionary
        int prompt_learning_start_id = reader.GetInteger(model_name, "prompt_learning_start_id", end_id + 1);
        PromptLearningType prompt_learning_type = static_cast<PromptLearningType>(reader.GetInteger(model_name, "prompt_learning_type", 0));
        

        RequestInfo r;
        size_t           max_input_len = -1;
        // Read ids of request from file.
        std::string start_ids_dir;
        if (model_name == "gptj_6b")
            start_ids_dir = "../examples/cpp/gptj/start_ids.csv";
        else if (model_name.substr(0, 5) == "llama")
            start_ids_dir = "../examples/cpp/llama/start_ids.csv";

        fl_read_start_ids(per_batch_size,
                    &r.v_start_lengths,
                    &r.v_start_ids,
                    max_input_len,
                    end_id,
                    1,
                    start_ids_dir);
        int total_output_len = max_input_len + request_output_len;
        
        r.total_seq_len                   = total_seq_len;
        r.beam_width                    = beam_width;
        r.top_k                         = top_k;
        r.top_p                         = top_p;
        r.tensor_para_size              = tensor_para_size;
        r.pipeline_para_size            = pipeline_para_size;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        float randomFloat = dis(gen);
        
        r.temperature                   = randomFloat;
        r.repetition_penalty            = repetition_penalty;
        r.presence_penalty              = presence_penalty;
        r.len_penalty                   = len_penalty;
        r.min_length                    = min_length;
        r.request_batch_size            = per_batch_size;
        r.memory_len                    = memory_len;
        r.prompt_learning_start_id      = prompt_learning_start_id;
        r.prompt_learning_type          = prompt_learning_type;
        r.num_tasks                     = num_tasks;
        r.max_input_len                 = max_input_len;
        r.random_seed                   = random_seed;
        r.beam_search_diversity_rate    = beam_search_diversity_rate;
        r.start_id                      = start_id;
        r.end_id                        = end_id;

        r.defined_generation_len        = defined_generation_len;
        r.unique_task_id                = task_id;

        std::string bad_words_dir;
        std::string stop_words_dir;
        if (model_name == "gptj_6b") {
            bad_words_dir = "/home/yao.877/parallel_inference/projects/FasterTransformer_AR/FasterTransformer_MV2/examples/cpp/gptj/bad_words.csv";
            stop_words_dir = "/home/yao.877/parallel_inference/projects/FasterTransformer_AR/FasterTransformer_MV2/examples/cpp/gptj/stop_words.csv";
        }
        else if (model_name.substr(0, 5) == "llama") {
            bad_words_dir = "/home/yao.877/parallel_inference/projects/FasterTransformer_AR/FasterTransformer_MV2/examples/cpp/llama/bad_words.csv";
            stop_words_dir = "/home/yao.877/parallel_inference/projects/FasterTransformer_AR/FasterTransformer_MV2/examples/cpp/llama/stop_words.csv";
        }

        read_word_list(bad_words_dir, r.bad_words);
        read_word_list(stop_words_dir, r.stop_words);
        const size_t stop_words_len = r.stop_words.size() / 2;

        for (int i = 0; i < per_batch_size; i++) {
            r.tiled_stop_words.insert(r.tiled_stop_words.end(), r.stop_words.begin(), r.stop_words.end());
        }

        r.head_num = head_num;
        r.size_per_head = size_per_head;
        r.vocab_size = vocab_size;
        r.decoder_layers = decoder_layers;
        r.rotary_embedding_dim = rotary_embedding_dim;
        r.inter_size = inter_size;
        r.hidden_units = hidden_units;
        r.fixed_prompt_len = fixed_prompt_len;

        r.start_time       = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        const size_t generation_len = reader.GetInteger("request", "request_output_len");
        r.generation_len = generation_len;
        _queue.push(r);
    }

    bool get(RequestInfo &value) {
        return _queue.try_pop(value);
    }
};

}