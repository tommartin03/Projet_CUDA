NVCC    = nvcc
CXX     = g++
TARGET  = filters
SRC_DIR     = src
INC_DIR     = include
BUILD_DIR   = build

SRCS = $(SRC_DIR)/main.cu     \
       $(SRC_DIR)/boxblur.cu  \
       $(SRC_DIR)/sobel.cu    \
       $(SRC_DIR)/laplace.cu  \
       $(SRC_DIR)/gaussian.cu

OBJS = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(SRCS))

OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4 2>/dev/null || \
                        pkg-config --cflags --libs opencv)

NVCC_FLAGS  = -std=c++11          \
              -I$(INC_DIR)        \
              -O2                  \
              $(OPENCV_FLAGS)

ARCH_FLAGS  = -arch=sm_75

.PHONY: all clean run

all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) $^ -o $@
	@echo ">>> Compilation réussie : $(TARGET)"

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
	@echo ">>> Nettoyage effectué"

run: all
	./$(TARGET) images/test.jpg 16
