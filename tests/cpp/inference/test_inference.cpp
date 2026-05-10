#include <chessmoe/chess/fen.h>
#include <chessmoe/eval/evaluator.h>
#include <chessmoe/inference/tensor_layout.h>
#include <chessmoe/inference/tensorrt_evaluator.h>
#include <chessmoe/cuda/batching/batch_planner.h>
#include <chessmoe/cuda/batching/cuda_stream.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void require_near(double actual, double expected, double tolerance,
                  std::string_view message) {
  if (std::fabs(actual - expected) > tolerance) {
    throw std::runtime_error(std::string(message) + ": expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(actual));
  }
}

class FakeInferenceBackend final : public chessmoe::inference::IInferenceBackend {
 public:
  chessmoe::inference::TensorLayout layout() const override {
    return chessmoe::inference::TensorLayout::tiny_baseline();
  }

  chessmoe::inference::RawNetworkOutput infer(
      const chessmoe::inference::NetworkInputBatch& batch) override {
    last_batch_size = batch.batch_size;
    require(batch.features.size() == batch.batch_size *
                                         layout().input_elements_per_position(),
            "encoded input feature count matches layout");

    chessmoe::inference::RawNetworkOutput output;
    output.batch_size = batch.batch_size;
    output.policy_logits.assign(batch.batch_size * layout().policy_buckets, -10.0F);
    output.wdl_logits.assign(batch.batch_size * 3, 0.0F);
    output.moves_left.assign(batch.batch_size, 42.0F);

    for (std::size_t i = 0; i < batch.batch_size; ++i) {
      output.wdl_logits[i * 3 + 0] = 2.0F;
      output.wdl_logits[i * 3 + 1] = 1.0F;
      output.wdl_logits[i * 3 + 2] = -1.0F;
      output.policy_logits[i * layout().policy_buckets +
                           chessmoe::inference::policy_index_from_uci("e2e4")] = 5.0F;
      output.policy_logits[i * layout().policy_buckets +
                           chessmoe::inference::policy_index_from_uci("e2e5")] = 100.0F;
    }

    return output;
  }

  std::size_t last_batch_size{0};
};

double probability_sum(const chessmoe::eval::EvaluationResult& result) {
  double sum = 0.0;
  for (const auto& entry : result.policy) {
    sum += entry.probability;
  }
  return sum;
}

bool contains_policy_move(const chessmoe::eval::EvaluationResult& result,
                          std::string_view uci) {
  for (const auto& entry : result.policy) {
    if (entry.move.to_uci() == uci) {
      return true;
    }
  }
  return false;
}

void test_tiny_tensor_layout_is_stable() {
  const auto layout = chessmoe::inference::TensorLayout::tiny_baseline();

  require(layout.input_name == "board", "input tensor is named board");
  require(layout.policy_output_name == "policy_logits",
          "policy output name is stable");
  require(layout.wdl_output_name == "wdl_logits", "WDL output name is stable");
  require(layout.moves_left_output_name == "moves_left",
          "moves-left output name is stable");
  require(layout.channels == 18, "tiny baseline uses 18 input channels");
  require(layout.height == 8 && layout.width == 8, "tiny baseline uses 8x8 board");
  require(layout.policy_buckets == 20480, "policy bucket count matches Python");
  require(layout.input_elements_per_position() == 18 * 8 * 8,
          "input element count is NCHW per position");
}

void test_policy_index_mapping_matches_python_contract() {
  require(chessmoe::inference::policy_index_from_uci("a1a1") == 0,
          "from-to policy index starts at a1a1");
  require(chessmoe::inference::policy_index_from_uci("e2e4") == 796,
          "e2e4 index matches Python encoder");
  require(chessmoe::inference::policy_index_from_uci("e7e8q") == 7484,
          "queen promotion offset matches Python encoder");
  require(chessmoe::inference::policy_index_from_uci("e7e8n") == 19772,
          "knight promotion offset matches Python encoder");
}

void test_batch_planner_uses_dynamic_tail_for_match_play() {
  chessmoe::cuda::batching::BatchPlanner planner({
      chessmoe::cuda::batching::InferenceMode::MatchPlay,
      1,
      4,
      4,
      false,
  });

  const auto batches = planner.plan(6);

  require(batches.size() == 2, "match play splits requests into two batches");
  require(batches[0].valid_count == 4 && batches[0].padded_count == 4,
          "match play first batch uses max batch");
  require(batches[1].valid_count == 2 && batches[1].padded_count == 2,
          "match play tail batch is not padded");
}

void test_batch_planner_pads_self_play_to_fixed_large_batch() {
  chessmoe::cuda::batching::BatchPlanner planner({
      chessmoe::cuda::batching::InferenceMode::SelfPlay,
      8,
      8,
      8,
      true,
  });

  const auto batches = planner.plan(10);

  require(batches.size() == 2, "self-play splits requests into fixed batches");
  require(batches[0].valid_count == 8 && batches[0].padded_count == 8,
          "self-play full batch has no padding");
  require(batches[1].valid_count == 2 && batches[1].padded_count == 8,
          "self-play tail is padded to fixed batch");
}

void test_cuda_stream_view_tracks_external_stream_without_cuda_headers() {
  int placeholder = 0;
  chessmoe::cuda::batching::CudaStreamView stream(&placeholder, false);

  require(static_cast<bool>(stream), "stream view with handle converts to true");
  require(stream.get() == &placeholder, "stream view stores opaque handle");
  require(!stream.owns_stream(), "external stream is not owned");
}

void test_tensor_rt_evaluator_masks_policy_to_legal_moves() {
  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  auto backend = std::make_shared<FakeInferenceBackend>();
  chessmoe::inference::TensorRTEvaluator evaluator(
      backend, chessmoe::inference::TensorRTEvaluatorConfig{});

  const std::array requests{
      chessmoe::eval::EvaluationRequest::from_position(position)};
  const auto results = evaluator.evaluate_batch(requests);

  require(results.size() == 1, "evaluator returns one result per request");
  require(backend->last_batch_size == 1, "backend receives one encoded position");

  const auto& result = results.front();
  require(result.policy.size() == 20, "start position has 20 legal moves");
  require(contains_policy_move(result, "e2e4"), "legal move is present");
  require(!contains_policy_move(result, "e2e5"), "illegal high-logit move is masked");
  require_near(probability_sum(result), 1.0, 1e-12,
               "masked policy probabilities sum to one");
  require(result.value > 0.0 && result.value < 1.0,
          "WDL logits convert to bounded scalar value");
  require(result.moves_left.has_value(), "moves-left output is propagated");
}

}  // namespace

int main() {
  try {
    test_tiny_tensor_layout_is_stable();
    test_policy_index_mapping_matches_python_contract();
    test_batch_planner_uses_dynamic_tail_for_match_play();
    test_batch_planner_pads_self_play_to_fixed_large_batch();
    test_cuda_stream_view_tracks_external_stream_without_cuda_headers();
    test_tensor_rt_evaluator_masks_policy_to_legal_moves();
  } catch (const std::exception& e) {
    std::cerr << "inference_tests failed: " << e.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
