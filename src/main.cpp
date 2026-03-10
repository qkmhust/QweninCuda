#include "model.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>

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

struct CliArgs {
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
  int no_repeat_ngram_size = 0;
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
  float repetition_penalty = 1.1f;
  int dump_topk = 0;
  int dump_steps = 0;
};

void usage() {
  std::cout << "Usage:\n"
            << "  qwen_minimal --model weights.qmini --input-ids 151644,872,198 --max-new-tokens 32 "
              "--eos-id 151645 --temperature 0.8 --top-k 40 --top-p 0.95 --min-p 0.0 "
              "--temp-decay 1.0 --greedy-after -1 --no-repeat-ngram-size 0 "
              "--presence-penalty 0.0 --frequency-penalty 0.0 --repetition-penalty 1.1 "
              "--dump-topk 0 --dump-steps 0\n";
}

CliArgs parse_args(int argc, char** argv) {
  CliArgs args;

  auto need_value = [&](int i, const std::string& name) {
    if (i + 1 >= argc) {
      throw std::runtime_error("Missing value for arg: " + name);
    }
  };

  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--model") {
      need_value(i, k);
      args.model_path = argv[++i];
    } else if (k == "--input-ids") {
      need_value(i, k);
      args.input_ids_str = argv[++i];
    } else if (k == "--max-new-tokens") {
      need_value(i, k);
      args.max_new_tokens = std::stoi(argv[++i]);
    } else if (k == "--eos-id") {
      need_value(i, k);
      args.eos_id = std::stoi(argv[++i]);
    } else if (k == "--temperature") {
      need_value(i, k);
      args.temperature = std::stof(argv[++i]);
    } else if (k == "--top-k") {
      need_value(i, k);
      args.top_k = std::stoi(argv[++i]);
    } else if (k == "--top-p") {
      need_value(i, k);
      args.top_p = std::stof(argv[++i]);
    } else if (k == "--min-p") {
      need_value(i, k);
      args.min_p = std::stof(argv[++i]);
    } else if (k == "--temp-decay") {
      need_value(i, k);
      args.temp_decay = std::stof(argv[++i]);
    } else if (k == "--greedy-after") {
      need_value(i, k);
      args.greedy_after = std::stoi(argv[++i]);
    } else if (k == "--no-repeat-ngram-size") {
      need_value(i, k);
      args.no_repeat_ngram_size = std::stoi(argv[++i]);
    } else if (k == "--presence-penalty") {
      need_value(i, k);
      args.presence_penalty = std::stof(argv[++i]);
    } else if (k == "--frequency-penalty") {
      need_value(i, k);
      args.frequency_penalty = std::stof(argv[++i]);
    } else if (k == "--repetition-penalty") {
      need_value(i, k);
      args.repetition_penalty = std::stof(argv[++i]);
    } else if (k == "--dump-topk") {
      need_value(i, k);
      args.dump_topk = std::stoi(argv[++i]);
    } else if (k == "--dump-steps") {
      need_value(i, k);
      args.dump_steps = std::stoi(argv[++i]);
    } else if (k == "-h" || k == "--help") {
      usage();
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown arg: " + k);
    }
  }

  if (args.model_path.empty() || args.input_ids_str.empty()) {
    throw std::runtime_error("--model and --input-ids are required");
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    CliArgs args = parse_args(argc, argv);
    std::vector<int> input_ids = parse_ids(args.input_ids_str);
    if (input_ids.empty()) {
      throw std::runtime_error("input ids parsed empty");
    }

    QwenMiniModel model;
    if (!model.load(args.model_path)) {
      return 1;
    }

    float elapsed_ms = 0.0f;
    auto output_ids =
      model.generate(input_ids, args.max_new_tokens, args.eos_id, args.temperature, args.top_k,
                     args.top_p, args.min_p, args.temp_decay, args.greedy_after,
                     args.no_repeat_ngram_size, args.presence_penalty, args.frequency_penalty,
                     args.repetition_penalty, args.dump_topk, args.dump_steps, &elapsed_ms);

    int new_tokens = static_cast<int>(output_ids.size() - input_ids.size());
    float tps = (new_tokens > 0 && elapsed_ms > 0.0f) ? (1000.0f * new_tokens / elapsed_ms) : 0.0f;

    std::cout << "generated_ids=" << join_ids(output_ids) << "\n";
    std::cout << "elapsed_ms=" << elapsed_ms << " new_tokens=" << new_tokens << " tok_per_s=" << tps
              << "\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    usage();
    return 2;
  }
}
