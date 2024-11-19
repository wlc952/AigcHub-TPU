// sherpa-onnx/csrc/online-zipformer2-ctc-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-zipformer2-ctc-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <string>

#include "bmruntime_interface.h"
#include "onnx-to-bm.h"
#include "sherpa-onnx/csrc/cat.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/unbind.h"
#include "debug_utils/include/utils.h"

using namespace bmruntime;

namespace sherpa_onnx {

class OnlineZipformer2CtcModel::Impl {
 public:
  explicit Impl(const OnlineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
          Init(config.zipformer2_ctc.model);
  }

  std::vector<Ort::Value> Forward(Ort::Value features,
                                  std::vector<Ort::Value> states) {
    // std::vector<Ort::Value> inputs;
    // inputs.reserve(1 + states.size());

    // inputs.push_back(std::move(features));
    // for (auto &v : states) {
    //   inputs.push_back(std::move(v));
    // }
    // auto outputs = sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
    //                   output_names_ptr_.data(), output_names_ptr_.size());

    // GetTensorMutableRawData<T> -> void*

    // load input from ort values
    auto &in_mem0 = net_info->stages[0].input_mems[0];
    bm_memcpy_s2d(bm_handle, in_mem0, features.GetTensorMutableRawData());
    for(int i = 0; i < states.size()-1; i++) {
      auto &in_mem = net_info->stages[0].input_mems[i+1];
      bm_memcpy_s2d(bm_handle, in_mem, states[i].GetTensorMutableRawData());
    }

    int64_t* raw_data_ptr = static_cast<int64_t*>(states.back().GetTensorMutableRawData());
    int32_t process_len = static_cast<int32_t>(*raw_data_ptr);

    bm_memcpy_s2d(bm_handle, net_info->stages[0].input_mems[states.size()], &process_len);

#ifdef DUMP_TENSOR
    dump_net_to_file(bm_handle, net_info, "./debug/tpu_input"+std::to_string(forward_cnt)+".npz"); // forward_cnt
    // // Dump every Ort::Value in the inputs
    dump_mem_to_file(features, "./debug/ort_input"+std::to_string(forward_cnt)+".npz", "input_0");
    for(int i = 0; i < states.size(); i++) {
      dump_mem_to_file(states[i], "./debug/ort_input"+std::to_string(forward_cnt)+".npz", "input_"+std::to_string(i+1));
    }
#endif

    // net launch
    std::vector<bm_tensor_t> in_tensors(net_info->input_num);
    std::vector<bm_tensor_t> out_tensors(net_info->output_num);

    for (int i = 0; i < net_info->input_num; i++) {
      bmrt_tensor_with_device(
          &in_tensors[i], net_info->stages[0].input_mems[i],
          net_info->input_dtypes[i], net_info->stages[0].input_shapes[i]);
    }
    for (int i = 0; i < net_info->output_num; i++) {
      bmrt_tensor_with_device(
          &out_tensors[i], net_info->stages[0].output_mems[i],
          net_info->output_dtypes[i], net_info->stages[0].output_shapes[i]);
    }
    auto ret = bmrt_launch_tensor_ex(p_bmrt, net_info->name, in_tensors.data(),
                                    net_info->input_num, out_tensors.data(),
                                    net_info->output_num, true, false);
    assert(ret);
    bm_thread_sync(bm_handle);

#ifdef DUMP_TENSOR
    for(int i = 0; i < _out_num; i++) {
      dump_tensor_to_file(bm_handle, out_tensors[i], net_info->stages[0].output_shapes[i], "./debug/tpu_output"+std::to_string(forward_cnt)+".npz",
                         BM_FLOAT32,
                         "output_"+std::to_string(i));
    }
#endif

    // load output to ort values
    std::vector<Ort::Value> outputs_ort;
    outputs_ort.reserve(net_info->output_num);

    Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    for(int i = 0; i < net_info->output_num-1; i++) {
      ONNXTensorElementDataType type;
      switch (net_info->output_dtypes[i]) {
        case 0:
          type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
          break;
        default:
          throw std::runtime_error("Unsupported data type for conversion.");
      }

      bm_shape_t tensor_shape = net_info->stages[0].output_shapes[i];
      int64_t *dim64 = new int64_t[tensor_shape.num_dims];
      // printf("%s ", net_info->output_names[i]);
      // printf("shape: ");
      int64_t num_elements = 1;
      for (int _i = 0; _i <  tensor_shape.num_dims; ++_i) {
        dim64[_i] = static_cast<int64_t>(tensor_shape.dims[_i]);
        // printf("%ld ", dim64[_i]);
        num_elements *= dim64[_i];
      }
      // printf(" == %ld \n", num_elements);
      
      void* temp_data_p = malloc(num_elements*sizeof(float));
      bm_memcpy_d2s(bm_handle, temp_data_p, net_info->stages[0].output_mems[i]);
      auto ort_value = Ort::Value::CreateTensor<float>(memory_info, (float*)temp_data_p, static_cast<size_t>(num_elements), dim64, tensor_shape.num_dims);
      // std::free(temp_data_p); #################

      outputs_ort.push_back(std::move(ort_value));
    }

    float process_len_float;
    bm_memcpy_d2s(bm_handle, &process_len_float, net_info->stages[0].output_mems[net_info->output_num-1]);
    int64_t process_len_int = static_cast<int64_t>(process_len_float);
    // printf("process_len: %ld\n", process_len_int);
    int64_t *dim64_1 = new int64_t[1];
    dim64_1[0] = 1;
    auto ort_value_1 = Ort::Value::CreateTensor<int64_t>(memory_info, &process_len_int, static_cast<size_t>(1), dim64_1, 1);
    std::free(dim64_1);

    outputs_ort.push_back(std::move(ort_value_1));

#ifdef DUMP_TENSOR
    for(int i = 0; i < _out_num; i++) {
      dump_mem_to_file(outputs_ort[i], "./debug/ort_output"+std::to_string(forward_cnt)+".npz", "output_"+std::to_string(i));
    }
#endif

    forward_cnt++;
    return outputs_ort;
  }

  int32_t VocabSize() const { return vocab_size_; }

  int32_t ChunkLength() const { return T_; }

  int32_t ChunkShift() const { return decode_chunk_len_; }

  OrtAllocator *Allocator() const { return allocator_; }

  // Return a vector containing 3 tensors
  // - attn_cache
  // - conv_cache
  // - offset
  std::vector<Ort::Value> GetInitStates() {
    std::vector<Ort::Value> ans;
    ans.reserve(initial_states_.size());
    for (auto &s : initial_states_) {
      ans.push_back(View(&s));
    }
    return ans;
  }

  std::vector<Ort::Value> StackStates(
      std::vector<std::vector<Ort::Value>> states) const {
    int32_t batch_size = static_cast<int32_t>(states.size());
    int32_t num_encoders = static_cast<int32_t>(num_encoder_layers_.size());

    std::vector<const Ort::Value *> buf(batch_size);

    std::vector<Ort::Value> ans;
    int32_t num_states = static_cast<int32_t>(states[0].size());
    ans.reserve(num_states);

    for (int32_t i = 0; i != (num_states - 2) / 6; ++i) {
      {
        for (int32_t n = 0; n != batch_size; ++n) {
          buf[n] = &states[n][6 * i];
        }
        auto v = Cat(allocator_, buf, 1);
        ans.push_back(std::move(v));
      }
      {
        for (int32_t n = 0; n != batch_size; ++n) {
          buf[n] = &states[n][6 * i + 1];
        }
        auto v = Cat(allocator_, buf, 1);
        ans.push_back(std::move(v));
      }
      {
        for (int32_t n = 0; n != batch_size; ++n) {
          buf[n] = &states[n][6 * i + 2];
        }
        auto v = Cat(allocator_, buf, 1);
        ans.push_back(std::move(v));
      }
      {
        for (int32_t n = 0; n != batch_size; ++n) {
          buf[n] = &states[n][6 * i + 3];
        }
        auto v = Cat(allocator_, buf, 1);
        ans.push_back(std::move(v));
      }
      {
        for (int32_t n = 0; n != batch_size; ++n) {
          buf[n] = &states[n][6 * i + 4];
        }
        auto v = Cat(allocator_, buf, 0);
        ans.push_back(std::move(v));
      }
      {
        for (int32_t n = 0; n != batch_size; ++n) {
          buf[n] = &states[n][6 * i + 5];
        }
        auto v = Cat(allocator_, buf, 0);
        ans.push_back(std::move(v));
      }
    }

    {
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = &states[n][num_states - 2];
      }
      auto v = Cat(allocator_, buf, 0);
      ans.push_back(std::move(v));
    }

    {
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = &states[n][num_states - 1];
      }
      auto v = Cat<int64_t>(allocator_, buf, 0);
      ans.push_back(std::move(v));
    }
    return ans;
  }

  std::vector<std::vector<Ort::Value>> UnStackStates(
      std::vector<Ort::Value> states) const {
    int32_t m = std::accumulate(num_encoder_layers_.begin(),
                                num_encoder_layers_.end(), 0);
    assert(states.size() == m * 6 + 2);

    int32_t batch_size = states[0].GetTensorTypeAndShapeInfo().GetShape()[1];
    int32_t num_encoders = num_encoder_layers_.size();

    std::vector<std::vector<Ort::Value>> ans;
    ans.resize(batch_size);

    for (int32_t i = 0; i != m; ++i) {
      {
        auto v = Unbind(allocator_, &states[i * 6], 1);
        assert(v.size() == batch_size);

        for (int32_t n = 0; n != batch_size; ++n) {
          ans[n].push_back(std::move(v[n]));
        }
      }
      {
        auto v = Unbind(allocator_, &states[i * 6 + 1], 1);
        assert(v.size() == batch_size);

        for (int32_t n = 0; n != batch_size; ++n) {
          ans[n].push_back(std::move(v[n]));
        }
      }
      {
        auto v = Unbind(allocator_, &states[i * 6 + 2], 1);
        assert(v.size() == batch_size);

        for (int32_t n = 0; n != batch_size; ++n) {
          ans[n].push_back(std::move(v[n]));
        }
      }
      {
        auto v = Unbind(allocator_, &states[i * 6 + 3], 1);
        assert(v.size() == batch_size);

        for (int32_t n = 0; n != batch_size; ++n) {
          ans[n].push_back(std::move(v[n]));
        }
      }
      {
        auto v = Unbind(allocator_, &states[i * 6 + 4], 0);
        assert(v.size() == batch_size);

        for (int32_t n = 0; n != batch_size; ++n) {
          ans[n].push_back(std::move(v[n]));
        }
      }
      {
        auto v = Unbind(allocator_, &states[i * 6 + 5], 0);
        assert(v.size() == batch_size);

        for (int32_t n = 0; n != batch_size; ++n) {
          ans[n].push_back(std::move(v[n]));
        }
      }
    }

    {
      auto v = Unbind(allocator_, &states[m * 6], 0);
      assert(v.size() == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }
    {
      auto v = Unbind<int64_t>(allocator_, &states[m * 6 + 1], 0);
      assert(v.size() == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }

    return ans;
  }

 private:
  void Init(const std::string &model_path) {
    // request bm_handle
    bm_status_t status = bm_dev_request(&bm_handle, dev_id);
    assert(BM_SUCCESS == status);

    // create bmruntime
  // create bmruntime
    p_bmrt = bmrt_create(bm_handle);
    assert(NULL != p_bmrt);

    // load bmodel by file
    bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
    assert(true == ret);
    printf("Model setup is done.\n");

    net_info = bmrt_get_network_info(p_bmrt, "zipformer2_ctc");
    assert(NULL != net_info);

    // setup input and output
    assert(net_info->input_num == _in_num);
    assert(net_info->output_num == _out_num);

    // set meta data
    encoder_dims_ = {192, 256, 384, 512, 384, 256};
    query_head_dims_ = {32, 32, 32, 32, 32, 32};
    value_head_dims_ = {12, 12, 12, 12, 12, 12};
    num_heads_ = {4, 4, 4, 8, 4, 4};
    num_encoder_layers_ = {2, 2, 3, 4, 3, 2};
    cnn_module_kernels_ = {31, 31, 15, 15, 15, 31};
    left_context_len_ = {128, 64, 32, 16, 32, 64};
    T_ = 45;
    decode_chunk_len_ = 32;
    vocab_size_ = 2000;

    InitStates();
  }

  void InitStates() {
    int32_t n = static_cast<int32_t>(encoder_dims_.size());
    int32_t m = std::accumulate(num_encoder_layers_.begin(),
                                num_encoder_layers_.end(), 0);
    initial_states_.reserve(m * 6 + 2);

    for (int32_t i = 0; i != n; ++i) {
      int32_t num_layers = num_encoder_layers_[i];
      int32_t key_dim = query_head_dims_[i] * num_heads_[i];
      int32_t value_dim = value_head_dims_[i] * num_heads_[i];
      int32_t nonlin_attn_head_dim = 3 * encoder_dims_[i] / 4;

      for (int32_t j = 0; j != num_layers; ++j) {
        {
          std::array<int64_t, 3> s{left_context_len_[i], 1, key_dim};
          auto v =
              Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
          Fill(&v, 0);
          initial_states_.push_back(std::move(v));
        }

        {
          std::array<int64_t, 4> s{1, 1, left_context_len_[i],
                                   nonlin_attn_head_dim};
          auto v =
              Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
          Fill(&v, 0);
          initial_states_.push_back(std::move(v));
        }

        {
          std::array<int64_t, 3> s{left_context_len_[i], 1, value_dim};
          auto v =
              Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
          Fill(&v, 0);
          initial_states_.push_back(std::move(v));
        }

        {
          std::array<int64_t, 3> s{left_context_len_[i], 1, value_dim};
          auto v =
              Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
          Fill(&v, 0);
          initial_states_.push_back(std::move(v));
        }

        {
          std::array<int64_t, 3> s{1, encoder_dims_[i],
                                   cnn_module_kernels_[i] / 2};
          auto v =
              Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
          Fill(&v, 0);
          initial_states_.push_back(std::move(v));
        }

        {
          std::array<int64_t, 3> s{1, encoder_dims_[i],
                                   cnn_module_kernels_[i] / 2};
          auto v =
              Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
          Fill(&v, 0);
          initial_states_.push_back(std::move(v));
        }
      }
    }

    {
      std::array<int64_t, 4> s{1, 128, 3, 19};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      initial_states_.push_back(std::move(v));
    }

    {
      std::array<int64_t, 1> s{1};
      auto v =
          Ort::Value::CreateTensor<int64_t>(allocator_, s.data(), s.size());
      Fill<int64_t>(&v, 0);
      initial_states_.push_back(std::move(v));
    }
  }

 private:
  OnlineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  int dev_id = 0;
  bm_handle_t bm_handle;
  void *p_bmrt;
  const bm_net_info_t *net_info;
  int _in_num = 99;
  int _out_num = 99;

  std::vector<Ort::Value> initial_states_;

  std::vector<int32_t> encoder_dims_;
  std::vector<int32_t> query_head_dims_;
  std::vector<int32_t> value_head_dims_;
  std::vector<int32_t> num_heads_;
  std::vector<int32_t> num_encoder_layers_;
  std::vector<int32_t> cnn_module_kernels_;
  std::vector<int32_t> left_context_len_;

  int32_t T_ = 0;
  int32_t decode_chunk_len_ = 0;
  int32_t vocab_size_ = 0;
  int forward_cnt = 0;
};

OnlineZipformer2CtcModel::OnlineZipformer2CtcModel(
    const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OnlineZipformer2CtcModel::~OnlineZipformer2CtcModel() = default;

std::vector<Ort::Value> OnlineZipformer2CtcModel::Forward(
    Ort::Value x, std::vector<Ort::Value> states) const {
  return impl_->Forward(std::move(x), std::move(states));
}

int32_t OnlineZipformer2CtcModel::VocabSize() const {
  return impl_->VocabSize();
}

int32_t OnlineZipformer2CtcModel::ChunkLength() const {
  return impl_->ChunkLength();
}

int32_t OnlineZipformer2CtcModel::ChunkShift() const {
  return impl_->ChunkShift();
}

OrtAllocator *OnlineZipformer2CtcModel::Allocator() const {
  return impl_->Allocator();
}

std::vector<Ort::Value> OnlineZipformer2CtcModel::GetInitStates() const {
  return impl_->GetInitStates();
}

std::vector<Ort::Value> OnlineZipformer2CtcModel::StackStates(
    std::vector<std::vector<Ort::Value>> states) const {
  return impl_->StackStates(std::move(states));
}

std::vector<std::vector<Ort::Value>> OnlineZipformer2CtcModel::UnStackStates(
    std::vector<Ort::Value> states) const {
  return impl_->UnStackStates(std::move(states));
}

}  // namespace sherpa_onnx