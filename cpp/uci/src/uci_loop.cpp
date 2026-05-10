#include <chessmoe/uci/uci_loop.h>

#include <mutex>
#include <string>
#include <thread>

namespace chessmoe::uci {

namespace {

bool starts_with(std::string_view text, std::string_view prefix) {
  return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

}  // namespace

UciLoop::UciLoop() = default;

void UciLoop::run(std::istream& input, std::ostream& output) {
  std::mutex output_mutex;
  std::thread search_thread;

  auto join_search = [&search_thread]() {
    if (search_thread.joinable()) {
      search_thread.join();
    }
  };

  std::string line;
  while (std::getline(input, line)) {
    if (starts_with(line, "go")) {
      join_search();
      search_thread = std::thread([this, line, &output, &output_mutex]() {
        const auto responses = engine_.handle_line(line);
        const std::lock_guard lock(output_mutex);
        for (const auto& response : responses) {
          output << response << '\n';
        }
        output.flush();
      });
      continue;
    }

    const auto responses = engine_.handle_line(line);
    {
      const std::lock_guard lock(output_mutex);
      for (const auto& response : responses) {
        output << response << '\n';
      }
      output.flush();
    }

    if (engine_.should_quit()) {
      join_search();
      break;
    }
  }

  join_search();
}

}  // namespace chessmoe::uci
