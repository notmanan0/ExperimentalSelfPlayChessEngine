#pragma once

#include <istream>
#include <ostream>

#include <chessmoe/uci/uci_engine.h>

namespace chessmoe::uci {

class UciLoop {
 public:
  UciLoop();

  void run(std::istream& input, std::ostream& output);

 private:
  UciEngine engine_;
};

}  // namespace chessmoe::uci
