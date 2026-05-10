#include <chessmoe/inference/tensorrt_evaluator.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace chessmoe::inference {
namespace {

void validate_output_shapes(const RawNetworkOutput& output,
                            const TensorLayout& layout,
                            std::size_t expected_batch) {
  if (output.batch_size != expected_batch) {
    throw std::runtime_error("backend returned wrong batch size");
  }
  if (output.policy_logits.size() != expected_batch * layout.policy_buckets) {
    throw std::runtime_error("backend returned wrong policy tensor size");
  }
  if (output.wdl_logits.size() != expected_batch * 3) {
    throw std::runtime_error("backend returned wrong WDL tensor size");
  }
  if (output.moves_left.size() != expected_batch) {
    throw std::runtime_error("backend returned wrong moves-left tensor size");
  }
}

std::array<double, 3> softmax3(const float* logits) {
  const float max_logit = std::max({logits[0], logits[1], logits[2]});
  std::array<double, 3> probabilities{
      std::exp(static_cast<double>(logits[0] - max_logit)),
      std::exp(static_cast<double>(logits[1] - max_logit)),
      std::exp(static_cast<double>(logits[2] - max_logit)),
  };
  const double sum = probabilities[0] + probabilities[1] + probabilities[2];
  for (auto& probability : probabilities) {
    probability /= sum;
  }
  return probabilities;
}

}  // namespace

TensorRTEvaluator::TensorRTEvaluator(std::shared_ptr<IInferenceBackend> backend,
                                     TensorRTEvaluatorConfig config)
    : backend_(std::move(backend)), config_(config) {
  if (!backend_) {
    throw std::invalid_argument("backend must not be null");
  }
}

std::vector<eval::EvaluationResult> TensorRTEvaluator::evaluate_batch(
    std::span<const eval::EvaluationRequest> requests) {
  if (requests.empty()) {
    return {};
  }

  const auto layout = backend_->layout();
  NetworkInputBatch input;
  input.batch_size = requests.size();
  input.features.reserve(requests.size() * layout.input_elements_per_position());
  for (const auto& request : requests) {
    auto encoded = encode_position_nchw(request.position, layout);
    input.features.insert(input.features.end(), encoded.begin(), encoded.end());
  }

  const auto output = backend_->infer(input);
  validate_output_shapes(output, layout, requests.size());

  std::vector<eval::EvaluationResult> results;
  results.reserve(requests.size());
  for (std::size_t batch_index = 0; batch_index < requests.size(); ++batch_index) {
    eval::EvaluationResult result;

    const auto* wdl = output.wdl_logits.data() + batch_index * 3;
    const auto wdl_probabilities = softmax3(wdl);
    result.wdl = {wdl_probabilities[0], wdl_probabilities[1],
                  wdl_probabilities[2]};
    result.value = result.wdl.win - result.wdl.loss;
    result.moves_left = output.moves_left[batch_index];

    const auto policy_offset = batch_index * layout.policy_buckets;
    for (const auto move : requests[batch_index].legal_moves) {
      const auto policy_index = policy_index_from_move(move);
      if (policy_index >= layout.policy_buckets) {
        throw std::runtime_error("legal move policy index exceeds layout");
      }
      result.policy.push_back(
          {move, output.policy_logits[policy_offset + policy_index], 0.0});
    }

    if (config_.apply_legal_mask) {
      result =
          eval::normalize_policy_over_legal_moves(std::move(result),
                                                  requests[batch_index].legal_moves);
    }
    results.push_back(std::move(result));
  }

  return results;
}

}  // namespace chessmoe::inference
