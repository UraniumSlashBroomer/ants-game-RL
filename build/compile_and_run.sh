g++ ../cpp_code/src/main.cpp \
    ../cpp_code/src/game_generation.cpp \
    ../cpp_code/src/render.cpp \
    -o main \
    -lraylib -lm -ldl -lpthread -lGL -lX11 \
    -I /opt/libtorch/include \
    -I /opt/libtorch/include/torch/csrc/api/include \
    -L /opt/libtorch/lib \
    -ltorch -ltorch_cpu -lc10 \
    -Wl,-rpath,/opt/libtorch/lib \
    -std=c++17 
./main
