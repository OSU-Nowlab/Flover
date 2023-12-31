# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)

add_library(GptJDecoderLayerWeight STATIC GptJDecoderLayerWeight.cc)
set_property(TARGET GptJDecoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptJDecoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptJDecoderLayerWeight PUBLIC memory_utils cuda_utils logger)

add_library(GptJDecoder STATIC GptJDecoder.cc)
set_property(TARGET GptJDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptJDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptJDecoder PUBLIC -lcudart cublasMMWrapper
                      TensorParallelDecoderSelfAttentionLayer
                      TensorParallelGeluFfnLayer
                      layernorm_kernels
                      add_residual_kernels
                      GptJDecoderLayerWeight
                      tensor
                      nccl_utils
                      cuda_utils
                      logger)

add_library(GptJContextDecoder STATIC GptJContextDecoder.cc)
set_property(TARGET GptJContextDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptJContextDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptJContextDecoder PUBLIC -lcudart cublasMMWrapper
                      TensorParallelGptContextAttentionLayer
                      TensorParallelGeluFfnLayer
                      layernorm_kernels
                      add_residual_kernels
                      gpt_kernels
                      tensor
                      nccl_utils
                      cuda_utils
                      logger)


add_library(GptJWeight STATIC GptJWeight.cc)
set_property(TARGET GptJWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptJWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptJWeight PUBLIC GptJDecoderLayerWeight cuda_utils logger)

add_library(GptJ STATIC GptJ.cc)
set_property(TARGET GptJ PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptJ PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptJ PUBLIC -lcudart
                      GptJDecoder
                      GptJContextDecoder
                      decoding_kernels
                      gpt_kernels
                      DynamicDecodeLayer
                      BaseBeamSearchLayer
                      bert_preprocess_kernels
                      tensor
                      GptJWeight
                      cuda_utils
                      logger)

add_library(GptJPreprocessing STATIC gptj_preprocessing.cc)
set_property(TARGET GptJPreprocessing PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptJPreprocessing PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptJPreprocessing PUBLIC -lcudart cublasMMWrapper
                    TensorParallelGptContextAttentionLayer
                    TensorParallelGeluFfnLayer
                    layernorm_kernels
                    add_residual_kernels
                    GptJDecoder
                    GptJContextDecoder
                    decoding_kernels
                    gpt_kernels
                    DynamicDecodeLayer
                    BaseBeamSearchLayer
                    bert_preprocess_kernels
                    tensor
                    GptJWeight
                    nccl_utils
                    cuda_utils
                    request_queue
                    logger)

add_library(GptJInference STATIC gptj_inference.cc)
set_property(TARGET GptJInference PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptJInference PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptJInference PUBLIC -lcudart cublasMMWrapper
                    TensorParallelGptContextAttentionLayer
                    TensorParallelGeluFfnLayer
                    layernorm_kernels
                    add_residual_kernels
                    GptJDecoder
                    GptJContextDecoder
                    decoding_kernels
                    gpt_kernels
                    DynamicDecodeLayer
                    BaseBeamSearchLayer
                    bert_preprocess_kernels
                    tensor
                    GptJWeight
                    nccl_utils
                    cuda_utils
                    request_queue
                    logger
                    GptJPreprocessing)