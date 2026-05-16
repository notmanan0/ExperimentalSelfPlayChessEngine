#include <chessmoe/chess/fen.h>
#include <chessmoe/chess/move_generator.h>
#include <chessmoe/inference/tensor_layout.h>

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void print_usage() {
  std::cerr
      << "usage: compat_helper <command> [args]\n"
      << "commands:\n"
      << "  policy-index <uci>            print policy bucket index for a UCI "
         "move\n"
      << "  encode-fen <fen>              print 18x8x8 encoded tensor as "
         "space-separated floats\n"
      << "  encode-fen-checksum <fen>     print adler32 checksum of encoded "
         "tensor\n"
      << "  legal-moves <fen>             print legal moves in UCI\n"
      << "  roundtrip-fen <fen>           parse and re-emit FEN\n";
}

std::uint32_t adler32(const float* data, std::size_t count) {
  constexpr std::uint32_t MOD = 65521;
  std::uint32_t a = 1;
  std::uint32_t b = 0;
  const auto* bytes = reinterpret_cast<const unsigned char*>(data);
  const std::size_t byte_count = count * sizeof(float);
  for (std::size_t i = 0; i < byte_count; ++i) {
    a = (a + bytes[i]) % MOD;
    b = (b + a) % MOD;
  }
  return (b << 16) | a;
}

int cmd_policy_index(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "policy-index requires a UCI move argument\n";
    return EXIT_FAILURE;
  }
  try {
    const auto index = chessmoe::inference::policy_index_from_uci(argv[2]);
    std::cout << index << '\n';
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int cmd_encode_fen(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "encode-fen requires a FEN argument\n";
    return EXIT_FAILURE;
  }
  try {
    const auto position = chessmoe::chess::Fen::parse(argv[2]);
    const auto layout = chessmoe::inference::TensorLayout::tiny_baseline();
    const auto encoded =
        chessmoe::inference::encode_position_nchw(position, layout);

    std::ostringstream out;
    out << std::fixed << std::setprecision(1);
    for (std::size_t i = 0; i < encoded.size(); ++i) {
      if (i > 0) {
        out << ' ';
      }
      out << encoded[i];
    }
    std::cout << out.str() << '\n';
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int cmd_encode_fen_checksum(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "encode-fen-checksum requires a FEN argument\n";
    return EXIT_FAILURE;
  }
  try {
    const auto position = chessmoe::chess::Fen::parse(argv[2]);
    const auto layout = chessmoe::inference::TensorLayout::tiny_baseline();
    const auto encoded =
        chessmoe::inference::encode_position_nchw(position, layout);

    const auto checksum = adler32(encoded.data(), encoded.size());
    std::cout << std::hex << std::setw(8) << std::setfill('0') << checksum
              << '\n';
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int cmd_legal_moves(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "legal-moves requires a FEN argument\n";
    return EXIT_FAILURE;
  }
  try {
    const auto position = chessmoe::chess::Fen::parse(argv[2]);
    const auto moves = chessmoe::chess::MoveGenerator::legal_moves(position);

    for (std::size_t i = 0; i < moves.size(); ++i) {
      if (i > 0) {
        std::cout << ' ';
      }
      std::cout << moves[i].to_uci();
    }
    std::cout << '\n';
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int cmd_roundtrip_fen(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "roundtrip-fen requires a FEN argument\n";
    return EXIT_FAILURE;
  }
  try {
    const auto position = chessmoe::chess::Fen::parse(argv[2]);
    std::cout << chessmoe::chess::Fen::to_string(position) << '\n';
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    return EXIT_FAILURE;
  }

  const std::string command = argv[1];

  if (command == "policy-index") {
    return cmd_policy_index(argc, argv);
  }
  if (command == "encode-fen") {
    return cmd_encode_fen(argc, argv);
  }
  if (command == "encode-fen-checksum") {
    return cmd_encode_fen_checksum(argc, argv);
  }
  if (command == "legal-moves") {
    return cmd_legal_moves(argc, argv);
  }
  if (command == "roundtrip-fen") {
    return cmd_roundtrip_fen(argc, argv);
  }

  std::cerr << "unknown command: " << command << '\n';
  print_usage();
  return EXIT_FAILURE;
}
