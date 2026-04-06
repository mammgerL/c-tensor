# C-Tensor: 轻量级张量计算库
# 支持 macOS (Accelerate) 和 Linux (OpenMP)

CC := clang
CFLAGS := -O3 -march=native -ffast-math -Wall -Wextra

# 检测操作系统
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    # macOS: 使用 Accelerate 框架
    LDFLAGS := -framework Accelerate
    # OpenMP 编译选项 (macOS)
    OMP_CFLAGS := -O3 -march=native -ffast-math -Wall -Wextra -DTENSOR_FORCE_OPENMP=1 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
    OMP_LDFLAGS := -L/opt/homebrew/opt/libomp/lib -lomp
else
    # Linux: 使用 OpenMP
    CC := gcc
    CFLAGS += -fopenmp
    LDFLAGS := -lm -fopenmp
    # Linux 下 OpenMP 就是默认选项
    OMP_CFLAGS := $(CFLAGS)
    OMP_LDFLAGS := $(LDFLAGS)
endif

# 目标文件
TARGETS := train eval

# 默认目标
all: $(TARGETS)

# 编译训练程序
train: train.c tensor.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# 编译评估程序
eval: eval.c tensor.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# OpenMP 版本编译 (macOS 上使用 libomp)
openmp: clean
	$(CC) $(OMP_CFLAGS) -o train train.c $(OMP_LDFLAGS)
	$(CC) $(OMP_CFLAGS) -o eval eval.c $(OMP_LDFLAGS)

# 运行完整流程：准备数据 -> 训练 -> 评估
run: data train eval
	./train
	./eval

# 准备 MNIST 数据集
data: mnist_train.csv mnist_test.csv

mnist_train.csv mnist_test.csv: create_mnist_csv.py
	python3 create_mnist_csv.py

# 准备 EMNIST Digits 数据集（独立于 MNIST）
EMNIST_RAW_DIR := data/EMNIST/digits/raw
EMNIST_BASE_URL := https://huggingface.co/datasets/Royc30ne/emnist-digits/resolve/main
EMNIST_RAW_FILES := \
	$(EMNIST_RAW_DIR)/emnist-digits-train-images-idx3-ubyte.gz \
	$(EMNIST_RAW_DIR)/emnist-digits-train-labels-idx1-ubyte.gz \
	$(EMNIST_RAW_DIR)/emnist-digits-test-images-idx3-ubyte.gz \
	$(EMNIST_RAW_DIR)/emnist-digits-test-labels-idx1-ubyte.gz

emnist-raw: $(EMNIST_RAW_FILES)

$(EMNIST_RAW_DIR)/emnist-digits-train-images-idx3-ubyte.gz:
	mkdir -p $(EMNIST_RAW_DIR)
	curl --http1.1 -L --retry 8 --retry-delay 2 --retry-all-errors --connect-timeout 15 -o $@ '$(EMNIST_BASE_URL)/emnist-digits-train-images-idx3-ubyte.gz?download=true'

$(EMNIST_RAW_DIR)/emnist-digits-train-labels-idx1-ubyte.gz:
	mkdir -p $(EMNIST_RAW_DIR)
	curl --http1.1 -L --retry 8 --retry-delay 2 --retry-all-errors --connect-timeout 15 -o $@ '$(EMNIST_BASE_URL)/emnist-digits-train-labels-idx1-ubyte.gz?download=true'

$(EMNIST_RAW_DIR)/emnist-digits-test-images-idx3-ubyte.gz:
	mkdir -p $(EMNIST_RAW_DIR)
	curl --http1.1 -L --retry 8 --retry-delay 2 --retry-all-errors --connect-timeout 15 -o $@ '$(EMNIST_BASE_URL)/emnist-digits-test-images-idx3-ubyte.gz?download=true'

$(EMNIST_RAW_DIR)/emnist-digits-test-labels-idx1-ubyte.gz:
	mkdir -p $(EMNIST_RAW_DIR)
	curl --http1.1 -L --retry 8 --retry-delay 2 --retry-all-errors --connect-timeout 15 -o $@ '$(EMNIST_BASE_URL)/emnist-digits-test-labels-idx1-ubyte.gz?download=true'

emnist-data: emnist_digits_train.csv emnist_digits_test.csv

emnist_digits_train.csv emnist_digits_test.csv: create_emnist_csv.py emnist-raw
	python3 create_emnist_csv.py --root $(EMNIST_RAW_DIR)

EMNIST_MODEL := emnist_digits_mlp.bin
EMNIST_WEB_PREFIX := emnist_samples

# 生成独立的 playground 风格合成手写数据
PLAYGROUND_DATA_DIR := generated/playground_handwritten
PLAYGROUND_DATA_CSV := $(PLAYGROUND_DATA_DIR)/playground_handwritten_train.csv

playground-data: $(PLAYGROUND_DATA_CSV)

$(PLAYGROUND_DATA_CSV): generate_handwritten.py
	mkdir -p $(PLAYGROUND_DATA_DIR)
	python3 generate_handwritten.py --count 10000 --output $(PLAYGROUND_DATA_CSV) --preview --preview-dir $(PLAYGROUND_DATA_DIR)/preview

# 仅训练
train-run: train data
	./train

# 使用 EMNIST Digits 训练单独模型
emnist-train-run: train emnist-data
	./train emnist_digits_train.csv $(EMNIST_MODEL)

# 仅评估
eval-run: eval data
	./eval

# 在 EMNIST Digits 测试集上评估 EMNIST 专用模型
emnist-eval-run: eval emnist-data
	./eval $(EMNIST_MODEL) emnist_digits_test.csv

# 将 EMNIST Digits 测试集导出为 Explore 页面可加载的二进制样本
emnist-web-data: emnist_digits_test.csv generate_test_samples.py
	python3 generate_test_samples.py --input emnist_digits_test.csv --prefix $(EMNIST_WEB_PREFIX) --out-dir web-app/src/assets

# 清理编译产物
clean:
	rm -f $(TARGETS)

# 清理所有生成文件（包括数据和模型）
cleanall: clean
	rm -f mnist_train.csv mnist_test.csv mnist_mlp.bin
	rm -f emnist_digits_train.csv emnist_digits_test.csv $(EMNIST_MODEL)


# Debug 编译（带调试符号，无优化）
debug: CFLAGS := -g -O0 -Wall -Wextra -DDEBUG
debug: clean $(TARGETS)

# 显示帮助信息
help:
	@echo "C-Tensor Makefile 使用说明"
	@echo ""
	@echo "目标:"
	@echo "  all        - 编译 train 和 eval (默认，macOS 用 Accelerate)"
	@echo "  openmp     - 使用 OpenMP 编译 (macOS 需 brew install libomp)"
	@echo "  train      - 仅编译训练程序"
	@echo "  eval       - 仅编译评估程序"
	@echo "  data       - 生成 MNIST CSV 数据集"
	@echo "  emnist-raw - 下载 EMNIST Digits 原始 gzip 文件（仅 digits 子集）"
	@echo "  emnist-data - 生成 EMNIST Digits CSV 数据集"
	@echo "  playground-data - 生成独立的 playground 风格合成手写数据"
	@echo "  run        - 完整流程: 数据准备 + 训练 + 评估"
	@echo "  train-run  - 编译并运行训练"
	@echo "  eval-run   - 编译并运行评估"
	@echo "  emnist-train-run - 使用 EMNIST Digits 训练 emnist_digits_mlp.bin"
	@echo "  emnist-eval-run - 在 EMNIST Digits 测试集上评估 emnist_digits_mlp.bin"
	@echo "  emnist-web-data - 导出 Explore 页面可加载的 EMNIST 样本 bin 文件"
	@echo "  debug      - Debug 编译 (带调试符号)"
	@echo "  clean      - 清理编译产物"
	@echo "  cleanall   - 清理所有生成文件"
	@echo "  help       - 显示此帮助信息"
	@echo ""
	@echo "Web 前端开发:"
	@echo "  cd web-app && npm run dev    - 启动开发服务器 (http://localhost:5173)"
	@echo "  cd web-app && npm run build  - 构建生产版本"
	@echo ""
	@echo "环境: $(UNAME_S)"
	@echo "编译器: $(CC)"
	@echo "编译选项: $(CFLAGS)"
	@echo "链接选项: $(LDFLAGS)"

.PHONY: all openmp run data emnist-data playground-data train-run eval-run emnist-train-run emnist-eval-run emnist-web-data clean cleanall debug help
