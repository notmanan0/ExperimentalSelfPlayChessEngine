#include <iostream>

#include <chessmoe/uci/uci_loop.h>

int main() {
  chessmoe::uci::UciLoop loop;
  loop.run(std::cin, std::cout);
  return 0;
}
