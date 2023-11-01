#include "src/flover/models/flover/Flover.h"
#include "src/flover/utils/nccl_utils.h"
#include "src/flover/utils/nvtx_utils.h"
#include "src/flover/utils/word_list.h"

#include <iostream>
#include <chrono>
#include <functional>
#include <vector>
#include <random>



template<typename T>
void flover_runtime(const INIReader& reader, int setDeviceNo, int rank)
{

    flover::Allocator<flover::AllocatorType::CUDA> _allocator(flover::getDevice());
    flover::Flover<T> flover_instance(reader, &_allocator);
    int stop_inference = 0;

    std::promise<void> p_fetching;
    std::future<void> f_fetching = p_fetching.get_future();
    std::promise<void> p_inference;
    std::future<void> f_inference = p_inference.get_future();


    flover_instance.serve(&stop_inference, setDeviceNo, std::ref(p_fetching), std::ref(p_inference));

    
    size_t max_concurrency = reader.GetInteger("model_specification", "max_concurrency");
    size_t max_seq_len     = reader.GetInteger("model_specification", "max_seq_len");
    int    interval        = reader.GetInteger("runtime_hyperparameter", "interval");
    int    use_mem_shuffle = reader.GetInteger("runtime_hyperparameter", "use_mem_shuffle");

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> random_lengths;

    int random_length = 0;
    for (int i=0;i<max_concurrency;++i) {
        if (use_mem_shuffle) {
            if(rank==0){
                std::random_device rd;
                std::mt19937 rng(rd());
                std::uniform_int_distribution<int> dist(128, max_seq_len);
                random_length = dist(rng);
                random_lengths.push_back(random_length);
            }
            MPI_Bcast(&random_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        flover_instance.addRequest(random_length);
        std::this_thread::sleep_for(std::chrono::milliseconds(interval));
    }
    
    stop_inference = 1;
    f_inference.get();
    f_fetching.get();
    flover::mpi::barrier();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    
    if (rank == 0) {
        if (use_mem_shuffle) {
            std::cout<<"Random request length:"<<std::endl;
            for (int i=0;i<random_lengths.size();++i) {
                std::cout<<" "<<random_lengths[i];
            }
            std::cout<<std::endl;
        }
        std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;
    }

    flover_instance.freeBuffer();
    // fprintf(stdout, "I am here\n");
}


int main(int argc, char* argv[]) {


    flover::mpi::initialize(&argc, &argv);
    
    // Prepare the parallelism parameters
    int rank       = flover::mpi::getCommWorldRank();
    int world_size = flover::mpi::getCommWorldSize();
    if (rank == 0) {
        printf("Total ranks: %d.\n", world_size);
    }
    int device, device_count;
    flover::check_cuda_error(cudaGetDeviceCount(&device_count));
    flover::check_cuda_error(cudaSetDevice(rank % device_count));
    flover::check_cuda_error(cudaGetDevice(&device));

    struct cudaDeviceProp prop;
    flover::check_cuda_error(cudaGetDeviceProperties(&prop, device));
    printf("Device %s\n", prop.name);

    printf("P%d is running with %d GPU.\n", rank, device);

    srand(0);

    std::string ini_name;
    if (argc == 2) {
        ini_name = std::string(argv[1]);
    }
    else {
        ini_name = "../src/flover/models/flover/configs/llama_config.ini";
    }

    INIReader reader = INIReader(ini_name);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << ini_name << "'\n";
        return -1;
    }
    const std::string model_type = reader.Get("model_specification", "model_type");
    const std::string data_type = reader.Get("model_specification", "data_type");

    if (data_type == "fp32") {
            flover_runtime<float>(reader, rank % device_count, rank);
        }
        else if (data_type == "fp16") {
            flover_runtime<half>(reader, rank % device_count, rank);
        }
    #ifdef ENABLE_BF16
        else if (data_type == "bf16") {
            flover_runtime<__nv_bfloat16>(reader, rank % device_count, rank);
        }
    #endif
        else {
            FT_LOG_ERROR("data_type should be fp32, fp16 or bf16!");
            return -1;
        }
    
    flover::mpi::finalize();
    return 0;
}