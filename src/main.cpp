#include "model.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<int> parse_ids(const std::string& s) {
  std::vector<int> ids;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      ids.push_back(std::stoi(item));
    }
  }
  return ids;
}

std::string join_ids(const std::vector<int>& ids) {
  std::ostringstream os;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (i > 0) {
      os << ',';
    }
    os << ids[i];
  }
  return os.str();
}

void usage() {
  std::cout << "Usage:\n"
            << "  qwen_minimal --model weights.qmini --input-ids 151644,872,198 --max-new-tokens 32 "
              "--eos-id 151645 --temperature 0.8 --top-k 40 --top-p 0.95 --min-p 0.0 "
              "--temp-decay 1.0 --greedy-after -1 --repetition-penalty 1.1\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string model_path;
    std::string input_ids_str;
    int max_new_tokens = 32;
    int eos_id = 151645;
    float temperature = 0.8f;
    int top_k = 40;
    float top_p = 0.95f;
    float min_p = 0.0f;
    float temp_decay = 1.0f;
    int greedy_after = -1;
    float repetition_penalty = 1.1f;

    for (int i = 1; i < argc; ++i) {
      std::string k = argv[i];
      if (k == "--model" && i + 1 < argc) {
        model_path = argv[++i];
      } else if (k == "--input-ids" && i + 1 < argc) {
        input_ids_str = argv[++i];
      } else if (k == "--max-new-tokens" && i + 1 < argc) {
        max_new_tokens = std::stoi(argv[++i]);
      } else if (k == "--eos-id" && i + 1 < argc) {
        eos_id = std::stoi(argv[++i]);
      } else if (k == "--temperature" && i + 1 < argc) {
        temperature = std::stof(argv[++i]);
      } else if (k == "--top-k" && i + 1 < argc) {
        top_k = std::stoi(argv[++i]);
      } else if (k == "--top-p" && i + 1 < argc) {
        top_p = std::stof(argv[++i]);
      } else if (k == "--min-p" && i + 1 < argc) {
        min_p = std::stof(argv[++i]);
      } else if (k == "--temp-decay" && i + 1 < argc) {
        temp_decay = std::stof(argv[++i]);
      } else if (k == "--greedy-after" && i + 1 < argc) {
        greedy_after = std::stoi(argv[++i]);
      } else if (k == "--repetition-penalty" && i + 1 < argc) {
        repetition_penalty = std::stof(argv[++i]);
      } else if (k == "-h" || k == "--help") {
        usage();
        return 0;
      } else {
        std::cerr << "Unknown arg: " << k << "\n";
        usage();
        return 1;
      }
    }

    if (model_path.empty() || input_ids_str.empty()) {
      usage();
      return 1;
    }

    std::vector<int> input_ids = parse_ids(input_ids_str);
    if (input_ids.empty()) {
      throw std::runtime_error("input ids parsed empty");
    }

    QwenMiniModel model;
    if (!model.load(model_path)) {
      return 1;
    }

    float elapsed_ms = 0.0f;
    auto output_ids =
      model.generate(input_ids, max_new_tokens, eos_id, temperature, top_k, top_p, min_p,
               temp_decay, greedy_after,
               repetition_penalty, &elapsed_ms);

    int new_tokens = static_cast<int>(output_ids.size() - input_ids.size());
    float tps = (new_tokens > 0 && elapsed_ms > 0.0f) ? (1000.0f * new_tokens / elapsed_ms) : 0.0f;

    std::cout << "generated_ids=" << join_ids(output_ids) << "\n";
    std::cout << "elapsed_ms=" << elapsed_ms << " new_tokens=" << new_tokens << " tok_per_s=" << tps
              << "\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    return 2;
  }
}
