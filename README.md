# C-Tensor

一个轻量级的纯 C 张量计算库，支持自动微分（Autograd），用于训练和评估 MNIST 手写数字识别神经网络。

## 项目特点

- **纯 C 实现**：无需 Python 运行时，仅依赖系统库
- **自动微分**：支持反向传播的计算图
- **跨平台优化**：
  - macOS：使用 Apple Accelerate 框架（vDSP + BLAS）
  - Linux：使用 OpenMP 并行化
- **高性能**：64 字节内存对齐，SIMD 友好
- **Header-Only**：核心库 `tensor.h` 单文件引入

## 项目结构

```
c-tensor/
├── tensor.h              # 核心张量库（Header-Only）
├── train.c               # MNIST 训练程序
├── eval.c                # 模型评估程序
├── create_mnist_csv.py   # 数据准备脚本
├── Makefile              # 构建配置
└── README.md             # 本文档
```

## 快速开始

### 1. 环境要求

**编译器**：
- macOS：Clang（系统自带）
- Linux：GCC 4.9+

**依赖**：
- macOS：Accelerate 框架（系统自带）
- Linux：OpenMP（通常已安装）

**数据准备**（Python）：
```bash
pip install torch torchvision numpy
```

### 2. 编译

```bash
# 编译所有程序
make

# 或分别编译
make train
make eval
```

### 3. 准备数据

```bash
# 生成 MNIST CSV 文件
make data
# 或
python3 create_mnist_csv.py
```

这将生成：
- `mnist_train.csv`：60,000 个训练样本
- `mnist_test.csv`：10,000 个测试样本

### 4. 训练模型

```bash
make train-run
# 或
./train
```

训练配置：
- 批大小：128
- 学习率：0.005
- 训练步数：20,000

训练完成后生成 `mnist_mlp.bin` 模型文件。

### 5. 评估模型

```bash
make eval-run
# 或
./eval
```

### 6. 一键运行

```bash
# 数据准备 → 训练 → 评估
make run
```

## 网络架构

```
输入层 (784) ──→ 隐藏层 (256) ──→ 输出层 (10)
           W1+B1         W2+B2
           ReLU        LogSoftmax
```

- **输入**：28×28 灰度图像展平为 784 维向量
- **隐藏层**：256 个神经元，ReLU 激活
- **输出层**：10 个类别（数字 0-9），LogSoftmax 归一化

## 支持的张量操作

| 操作 | 描述 | 前向 | 反向 |
|------|------|------|------|
| MATMUL | 矩阵乘法 | ✓ | ✓ |
| MEAN | 均值归约 | ✓ | ✓ |
| MUL | 逐元素乘法 | ✓ | ✓ |
| RELU | ReLU 激活 | ✓ | ✓ |
| LOGSOFTMAX | Log-Softmax | ✓ | ✓ |
| SUM_AXIS1 | 沿轴1求和 | ✓ | ✓ |
| ADD_BIAS | 偏置加法 | ✓ | ✓ |

## API 简介

### 数据结构

```c
// 数组容器
typedef struct {
    float *values;      // 数据指针
    int *shape;         // 形状
    int *strides;       // 步长
    int ndim;           // 维度数
    int size;           // 元素总数
} Arr;

// 可微分张量
typedef struct Tensor {
    Arr *data;          // 前向数据
    Arr *grad;          // 梯度数据
    int op;             // 操作码
    struct Tensor *lhs; // 左操作数
    struct Tensor *rhs; // 右操作数
    Arg arg;            // 操作参数
} Tensor;
```

### 核心函数

```c
// 创建张量
Tensor *tensor_create(int ndim, int *shape);
Tensor *tensor_from_arr(Arr *arr);

// 张量操作
Tensor *matmul(Tensor *lhs, Tensor *rhs);
Tensor *add_bias(Tensor *input, Tensor *bias);
Tensor *tensor_relu(Tensor *input);
Tensor *log_softmax(Tensor *input);
Tensor *tensor_mul(Tensor *lhs, Tensor *rhs);
Tensor *mean(Tensor *input);

// 自动微分
void backward(Tensor *t);
void zero_grad(Tensor *t);

// 模型持久化
void save_model(const char *path, Tensor *w1, Tensor *b1, Tensor *w2, Tensor *b2);
void load_model(const char *path, Tensor **w1, Tensor **b1, Tensor **w2, Tensor **b2);

// 初始化
void kaiming_uniform_(Arr *arr, int fan_in);
```

## Makefile 命令

| 命令 | 描述 |
|------|------|
| `make` | 编译 train 和 eval（macOS 用 Accelerate） |
| `make openmp` | 使用 OpenMP 编译（macOS 需先 `brew install libomp`） |
| `make train` | 仅编译训练程序 |
| `make eval` | 仅编译评估程序 |
| `make data` | 生成 MNIST 数据集 |
| `make run` | 完整流程：数据 → 训练 → 评估 |
| `make train-run` | 编译并运行训练 |
| `make train-ane` | 编译 ANE 实验训练程序 |
| `make train-ane-run` | 编译并运行 ANE 实验训练程序 |
| `make eval-run` | 编译并运行评估 |
| `make web` | 启动 Web 可视化服务 |
| `make debug` | Debug 编译（带调试符号） |
| `make clean` | 清理编译产物 |
| `make cleanall` | 清理所有生成文件 |
| `make help` | 显示帮助信息 |

## 性能基准测试

在 Apple Silicon (M 系列芯片) 上的性能对比：

| 指标 | Apple Accelerate | OpenMP | ANE（静态权重路径） | ANE（动态权重路径） |
|------|------------------|--------|---------------------|---------------------|
| **总训练时间** | **6.24 秒** | 163.00 秒 | 248.14 秒 | 7.59 秒 |
| **平均每步耗时** | **0.31 ms** | 8.15 ms | 12.41 ms | 0.38 ms |
| **测试准确率** | 95.66% | 95.66% | 95.66% | 95.67% |
| **相对 Accelerate** | 基准 | 约 26x 慢 | 约 40x 慢 | 约 1.22x 慢 |

> 测试环境：macOS, Apple Silicon, 20,000 训练步，批大小 128。ANE 为实验私有 API 路径（第一层前向在 ANE，反向仍在 CPU）。

### 结论

- **macOS 强烈推荐使用 Accelerate**（默认选项）
- Accelerate 使用 vDSP/BLAS 高度优化的向量化矩阵运算，针对 Apple Silicon 深度优化
- OpenMP 版本适合 Linux 环境或需要跨平台一致性的场景
- ANE 静态权重路径可跑通训练，但会因重复编译导致明显变慢
- ANE 动态权重路径（`ANE_DYNAMIC_WEIGHTS=1`）已避免每步重编译，速度接近 Accelerate，但当前仍未超过 Accelerate

## 性能优化

1. **内存对齐**：64 字节对齐分配，优化 SIMD 访问
2. **平台加速**：
   - macOS 使用 vDSP/BLAS 向量化运算（推荐）
   - Linux 使用 OpenMP 多线程并行
3. **高效 I/O**：1MB 文件缓冲，strtof 快速解析
4. **缓存友好**：Fisher-Yates 洗牌保持数据局部性

## 扩展使用

### ANE 实验训练（macOS，私有 API）

> 该路径是实验功能，默认关闭，不影响主线 Accelerate/OpenMP。

```bash
# 编译 ANE 实验训练程序
make train-ane

# 运行 ANE 模式（第一层 dense 前向走 ANE，反向仍在 CPU）
ANE_ENABLE_PRIVATE_API=1 TENSOR_USE_ANE=1 ./train_ane

# 运行推荐的动态权重模式（避免每步重编译）
ANE_ENABLE_PRIVATE_API=1 ANE_DYNAMIC_WEIGHTS=1 TENSOR_USE_ANE=1 ./train_ane

# 可选：缩短训练用于验证
ANE_ENABLE_PRIVATE_API=1 TENSOR_USE_ANE=1 TRAIN_STEPS=300 ./train_ane
```

可用环境变量：
- `TENSOR_USE_ANE=1`：启用训练循环中的 ANE 路径
- `ANE_ENABLE_PRIVATE_API=1`：启用私有 ANE API 探测与调用
- `ANE_DYNAMIC_WEIGHTS=1`：启用动态权重路径（推荐；避免每步重编译）
- `TENSOR_USE_ANE_LAYER2`：第二层后端策略（`1` 强制 ANE，`0` 强制 CPU，默认 `-1` 自动）
- `TENSOR_ANE_LAYER2_MIN_MACS`：第二层自动启用 ANE 的 MAC 阈值（默认 `1000000`）
- `TRAIN_STEPS` / `TRAIN_BATCH` / `TRAIN_LR`：覆盖默认训练参数

自动策略说明：
- 默认会根据第二层形状自动选后端，避免小矩阵走 ANE 导致变慢。
- 当前默认配置下（`B=128, H=256, O=10`）会自动选择 CPU。

### 使用 OpenMP 编译（macOS）

```bash
# 安装 libomp
brew install libomp

# 使用 OpenMP 编译
make openmp
```

### 自定义网络

修改 `train.c` 中的网络参数：

```c
#define BATCH_SIZE 128      // 批大小
#define LEARNING_RATE 0.005 // 学习率
#define TRAIN_STEPS 20000   // 训练步数
#define HIDDEN_SIZE 256     // 隐藏层大小
```

## Web 可视化服务

提供交互式 Web 界面，用于：
- 浏览测试样本（支持按正确/错误过滤）
- 查看识别错误的图片详情
- 展示分类概率分布
- 可视化隐藏层激活
- 查看网络权重（W1 滤波器、W2 连接）

### 启动服务

```bash
# 确保已训练模型和生成数据
make data
make train-run

# 启动 Web 服务
make web
```

访问 http://localhost:3000

### 功能截图

- **样本网格**：缩略图展示，绿色标记正确，红色标记错误
- **详情视图**：放大图片、真实/预测标签、概率条形图、激活热力图
- **权重可视化**：256 个隐藏层神经元的 28x28 滤波器

## 模型文件格式

`mnist_mlp.bin` 二进制格式：

```
[4 bytes] Magic: "MLP1"
[4 bytes] W1 行数
[4 bytes] W1 列数
[N floats] W1 数据
[4 bytes] B1 大小
[N floats] B1 数据
[4 bytes] W2 行数
[4 bytes] W2 列数
[N floats] W2 数据
[4 bytes] B2 大小
[N floats] B2 数据
```

## 许可证

MIT License
