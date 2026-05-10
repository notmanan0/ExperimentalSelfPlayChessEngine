#include <chessmoe/chess/fen.h>
#include <chessmoe/chess/perft.h>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

namespace {

void print_usage() {
  std::cerr << "usage: chessmoe_perft \"<fen>\" <depth> [--divide]\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    print_usage();
    return EXIT_FAILURE;
  }

  try {
    const std::string fen = argv[1];
    const int depth = std::stoi(argv[2]);
    const bool divide = argc == 4 && std::string{argv[3]} == "--divide";
    const auto position = chessmoe::chess::Fen::parse(fen);

    if (divide) {
      std::uint64_t total = 0;
      for (const auto& [move, nodes] : chessmoe::chess::perft_divide(position, depth)) {
        std::cout << move << ": " << nodes << '\n';
        total += nodes;
      }
      std::cout << "nodes: " << total << '\n';
    } else {
      std::cout << chessmoe::chess::perft(position, depth) << '\n';
    }
  } catch (const std::exception& e) {
    std::cerr << "perft failed: " << e.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
