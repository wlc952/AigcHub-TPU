// sherpa-onnx/csrc/online-zipformer2-ctc-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-zipformer2-ctc-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <string>

#include "bmruntime_cpp.h"
#include "onnx-to-bm.h"
#include "sherpa-onnx/csrc/cat.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/unbind.h"
// #include "debug_utils/include/utils.h"

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
    std::vector<Ort::Value> inputs;
    inputs.reserve(1 + states.size());

    inputs.push_back(std::move(features));
    for (auto &v : states) {
      inputs.push_back(std::move(v));
    }

    auto &bm_inputs = _net->Inputs();
    auto &bm_outputs = _net->Outputs();

    LoadOrtValuesToBMTensors(inputs, bm_inputs, _in_num);

    bool status = _net->Forward();

    assert(BM_SUCCESS == status);
    std::vector<Ort::Value> outputs_ort = GetOrtValuesFromBMTensors(bm_outputs.begin(), _out_num-1);
    
    Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    float temp = 0.0f;
    bm_status_t status_tmp = bm_outputs[_out_num-1]->CopyTo((void*)(&temp));
    assert(BM_SUCCESS == status_tmp);
    int64_t intValue = static_cast<int64_t>(temp);
    std::vector<int64_t> shape_tmp = {1};
    Ort::Value lastOrtValue = Ort::Value::CreateTensor<int64_t>(memory_info, &intValue, 1, shape_tmp.data(), 1);
    outputs_ort.push_back(std::move(lastOrtValue)); // lastOrtValue);

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
    _ctx = std::make_shared<Context>(dev_id);
    bm_status_t status = _ctx->load_bmodel(model_path.c_str());
    assert(BM_SUCCESS == status);

    // create Network
    std::vector<const char *> network_names;
    _ctx->get_network_names(&network_names);
    _net = std::make_shared<Network>(*_ctx, network_names[0], 0);  // use stage[0]
    // SHERPA_ONNX_LOGE("model_in out: %d %d", _net->info()->input_num, _net->info()->output_num);
    assert(_net->info()->input_num == _in_num &&
           _net->info()->output_num == _out_num);

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

  std::shared_ptr<bmruntime::Context> _ctx;
  std::shared_ptr<bmruntime::Network> _net;
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