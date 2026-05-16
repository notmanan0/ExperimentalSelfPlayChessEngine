add_executable(compat_helper main.cpp)

target_link_libraries(compat_helper
  PRIVATE
    chessmoe_chess
    chessmoe_eval
    chessmoe_inference
)

target_compile_features(compat_helper PRIVATE cxx_std_20)
