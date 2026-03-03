# C-Tensor: 轻量级张量计算库
# 支持 macOS (Accelerate) 和 Linux (OpenMP)

CC := clang
CFLAGS := -O3 -march=native -ffast-math -Wall -Wextra
PYTHON ?= python3

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
TARGETS := train eval web

# 默认目标
all: $(TARGETS)

# 编译训练程序
train: train.c tensor.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# 编译评估程序
eval: eval.c tensor.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# 编译 Web 服务器
web: web_server.c api_handlers.c tensor.h
	$(CC) $(CFLAGS) -o $@ web_server.c api_handlers.c $(LDFLAGS)

# 编译 ANE M1 PoC（实验分支，不进入默认 all）
ane-poc: ane_poc.c ane_backend.h ane_backend.m tensor.h
	$(CC) $(CFLAGS) -o ane_poc ane_poc.c ane_backend.m $(LDFLAGS) -lobjc

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
	$(PYTHON) create_mnist_csv.py

# 仅训练
train-run: train data
	./train

# 仅评估
eval-run: eval data
	./eval

# 启动 Web 服务
web-run: web data
	./web

# 运行 ANE M1 PoC
ane-poc-run: ane-poc data
	./ane_poc

# 清理编译产物
clean:
	rm -f $(TARGETS) ane_poc bench_accelerate.log bench_openmp.log

# 清理所有生成文件（包括数据和模型）
cleanall: clean
	rm -f mnist_train.csv mnist_test.csv mnist_mlp.bin


# Debug 编译（带调试符号，无优化）
debug: CFLAGS := -g -O0 -Wall -Wextra -DDEBUG
debug: clean $(TARGETS)

# 显示帮助信息
help:
	@echo "C-Tensor Makefile 使用说明"
	@echo ""
	@echo "目标:"
	@echo "  all        - 编译 train、eval 和 web (默认，macOS 用 Accelerate)"
	@echo "  openmp     - 使用 OpenMP 编译 (macOS 需 brew install libomp)"
	@echo "  train      - 仅编译训练程序"
	@echo "  eval       - 仅编译评估程序"
	@echo "  web        - 仅编译 Web 服务器"
	@echo "  data       - 生成 MNIST CSV 数据集"
	@echo "  run        - 完整流程: 数据准备 + 训练 + 评估"
	@echo "  train-run  - 编译并运行训练"
	@echo "  eval-run   - 编译并运行评估"
	@echo "  web-run    - 编译并启动 Web 服务"
	@echo "  ane-poc    - 编译 ANE M1 前向 PoC 程序"
	@echo "  ane-poc-run - 运行 ANE M1 前向 PoC (含数据准备)"
	@echo "  debug      - Debug 编译 (带调试符号)"
	@echo "  bench      - 跑 Accelerate vs OpenMP 训练耗时对比"
	@echo "  clean      - 清理编译产物"
	@echo "  cleanall   - 清理所有生成文件"
	@echo "  help       - 显示此帮助信息"
	@echo ""
	@echo "环境: $(UNAME_S)"
	@echo "编译器: $(CC)"
	@echo "编译选项: $(CFLAGS)"
	@echo "链接选项: $(LDFLAGS)"

bench:
	./scripts/bench_backends.sh

.PHONY: all openmp run data train-run eval-run web-run ane-poc ane-poc-run clean cleanall debug help bench
