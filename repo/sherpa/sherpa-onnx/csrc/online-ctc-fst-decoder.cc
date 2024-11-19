// sherpa-onnx/csrc/online-ctc-fst-decoder.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-ctc-fst-decoder.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fst/fstlib.h"
#include "kaldi-decoder/csrc/decodable-ctc.h"
#include "kaldifst/csrc/fstext-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-stream.h"

namespace sherpa_onnx {

// This function is copied from kaldi.
//
// @param filename Path to a StdVectorFst or StdConstFst graph
// @return The caller should free the returned pointer using `delete` to
//         avoid memory leak.
fst::Fst<fst::StdArc> *ReadGraph(const std::string &filename) {
  // read decoding network FST
  std::ifstream is(filename, std::ios::binary);
  if (!is.good()) {
    SHERPA_ONNX_LOGE("Could not open decoding-graph FST %s", filename.c_str());
  }

  fst::FstHeader hdr;
  if (!hdr.Read(is, "<unknown>")) {
    SHERPA_ONNX_LOGE("Reading FST: error reading FST header.");
  }

  if (hdr.ArcType() != fst::StdArc::Type()) {
    SHERPA_ONNX_LOGE("FST with arc type %s not supported",
                     hdr.ArcType().c_str());
  }
  fst::FstReadOptions ropts("<unspecified>", &hdr);

  fst::Fst<fst::StdArc> *decode_fst = nullptr;

  if (hdr.FstType() == "vector") {
    decode_fst = fst::VectorFst<fst::StdArc>::Read(is, ropts);
  } else if (hdr.FstType() == "const") {
    decode_fst = fst::ConstFst<fst::StdArc>::Read(is, ropts);
  } else {
    SHERPA_ONNX_LOGE("Reading FST: unsupported FST type: %s",
                     hdr.FstType().c_str());
  }

  if (decode_fst == nullptr) {  // fst code will warn.
    SHERPA_ONNX_LOGE("Error reading FST (after reading header).");
    return nullptr;
  } else {
    return decode_fst;
  }
}

OnlineCtcFstDecoder::OnlineCtcFstDecoder(
    const OnlineCtcFstDecoderConfig &config, int32_t blank_id)
    : config_(config), fst_(ReadGraph(config.graph)), blank_id_(blank_id) {
  options_.max_active = config_.max_active;
}

std::unique_ptr<kaldi_decoder::FasterDecoder>
OnlineCtcFstDecoder::CreateFasterDecoder() const {
  return std::make_unique<kaldi_decoder::FasterDecoder>(*fst_, options_);
}

static void DecodeOne(const float *log_probs, int32_t num_rows,
                      int32_t num_cols, OnlineCtcDecoderResult *result,
                      OnlineStream *s, int32_t blank_id) {
  int32_t &processed_frames = s->GetFasterDecoderProcessedFrames();
  kaldi_decoder::DecodableCtc decodable(log_probs, num_rows, num_cols); //, processed_frames);

  kaldi_decoder::FasterDecoder *decoder = s->GetFasterDecoder();
  if (processed_frames == 0) {
    decoder->InitDecoding();
  }

  decoder->AdvanceDecoding(&decodable);

  if (decoder->ReachedFinal()) {
    fst::VectorFst<fst::LatticeArc> fst_out;
    bool ok = decoder->GetBestPath(&fst_out);
    if (ok) {
      std::vector<int32_t> isymbols_out;
      std::vector<int32_t> osymbols_out_unused;
      ok = fst::GetLinearSymbolSequence(fst_out, &isymbols_out,
                                        &osymbols_out_unused, nullptr);
      std::vector<int64_t> tokens;
      tokens.reserve(isymbols_out.size());

      std::vector<int32_t> timestamps;
      timestamps.reserve(isymbols_out.size());

      std::ostringstream os;
      int32_t prev_id = -1;
      int32_t num_trailing_blanks = 0;
      int32_t f = 0;  // frame number

      for (auto i : isymbols_out) {
        i -= 1;

        if (i == blank_id) {
          num_trailing_blanks += 1;
        } else {
          num_trailing_blanks = 0;
        }

        if (i != blank_id && i != prev_id) {
          tokens.push_back(i);
          timestamps.push_back(f);
        }
        prev_id = i;
        f += 1;
      }

      result->tokens = std::move(tokens);
      result->timestamps = std::move(timestamps);
      // no need to set frame_offset
    }
  }

  processed_frames += num_rows;
}

void OnlineCtcFstDecoder::Decode(Ort::Value log_probs,
                                 std::vector<OnlineCtcDecoderResult> *results,
                                 OnlineStream **ss, int32_t n) {
  std::vector<int64_t> log_probs_shape =
      log_probs.GetTensorTypeAndShapeInfo().GetShape();

  if (log_probs_shape[0] != results->size()) {
    SHERPA_ONNX_LOGE("Size mismatch! log_probs.size(0) %d, results.size(0): %d",
                     static_cast<int32_t>(log_probs_shape[0]),
                     static_cast<int32_t>(results->size()));
    exit(-1);
  }

  if (log_probs_shape[0] != n) {
    SHERPA_ONNX_LOGE("Size mismatch! log_probs.size(0) %d, n: %d",
                     static_cast<int32_t>(log_probs_shape[0]), n);
    exit(-1);
  }

  int32_t batch_size = static_cast<int32_t>(log_probs_shape[0]);
  int32_t num_frames = static_cast<int32_t>(log_probs_shape[1]);
  int32_t vocab_size = static_cast<int32_t>(log_probs_shape[2]);

  const float *p = log_probs.GetTensorData<float>();

  for (int32_t i = 0; i != batch_size; ++i) {
    DecodeOne(p + i * num_frames * vocab_size, num_frames, vocab_size,
              &(*results)[i], ss[i], blank_id_);
  }
}

}  // namespace sherpa_onnx
