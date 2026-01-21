# AGENT DEVELOPMENT GUIDELINES

This document serves as the style guide, build instructions, and best practices for all automated agents operating within the `c-tensor` repository.

## 1. Project Context

- **Primary Language:** Pure C (`.c`, `.h`).
- **Core Library:** `tensor.h` is a header-only library implementing tensors and automatic differentiation.
- **Goal:** Maintain simplicity, performance, and cross-platform compatibility (macOS Accelerate/Linux OpenMP).
- **Style Classification:** Disciplined C style, prioritizing performance optimization (aligned memory, SIMD friendly).

## 2. Build, Lint, and Test Commands

### 2.1. Build & Run

The build process is managed by `Makefile`. Default compilation uses platform-specific optimizations.

| Command | Description | Notes |
| :--- | :--- | :--- |
| `make` | Compiles `train` and `eval` executables. | Default: Uses Accelerate on macOS, OpenMP on Linux/GCC. |
| `make openmp` | Compiles using OpenMP for cross-platform consistency. | Requires `libomp` on macOS (`brew install libomp`). |
| `make debug` | Compiles with `-g -O0 -DDEBUG` flags. | Use for any debugging tasks. |
| `make data` | Prepares the MNIST CSV dataset (requires Python dependencies). | Requires `pip install torch torchvision numpy` |

### 2.2. Training & Evaluation

| Command | Description | Notes |
| :--- | :--- | :--- |
| `make train-run` | Compiles and runs the training program (`./train`). | Generates `mnist_mlp.bin`. |
| `make eval-run` | Compiles and runs the evaluation program (`./eval`). | Reports final accuracy on test set. |
| `make run` | Full pipeline: Data preparation → Train → Eval. | Recommended for full workflow verification. |

### 2.3. Testing

- **Testing Framework:** None (no CUnit, GoogleTest, etc.).
- **Testing Approach:** Evaluation is performed via the dedicated `eval` executable.
- **Running Tests:** Run the full evaluation suite. **It is not possible to run a single, isolated unit test.**
  ```bash
  # Run the full test/evaluation suite (10,000 samples)
  ./eval
  ```

### 2.4. Linting and Formatting

No formal linter or formatter configuration (`.clang-format`, `.eslintrc`, etc.) was found. Code style must be inferred from the existing C files (`train.c`, `eval.c`, `tensor.h`).

## 3. Code Style Guidelines (Pure C)

### 3.1. Naming Conventions

| Entity | Convention | Examples |
| :--- | :--- | :--- |
| **Functions** | `snake_case` | `load_csv`, `sampler_init`, `free_tensor` |
| **Variables** | `snake_case` | `w1`, `batch_x`, `loss_val`, `curB` |
| **Types/Structs** | `PascalCase` (with `typedef struct`) | `Arr`, `Tensor`, `ModelHeader`, `BatchSampler` |
| **Macros/Constants** | `SCREAMING_SNAKE_CASE` | `MATMUL`, `MAX_PREVS`, `USE_ACCELERATE` |

### 3.2. Formatting

- **Indentation:** 4 spaces.
- **Braces (`{}`):** K&R style (opening brace on the same line as the statement).
  ```c
  static void some_function(int arg) {
      if (arg > 0) {
          // 4 spaces
      }
  }
  ```
- **Pointers:** Space before the asterisk, e.g., `Arr* data;`, NOT `Arr *data;`.

### 3.3. Memory Management (CRITICAL)

The project uses manual memory management (`malloc`/`free`). Agents MUST adhere to the following:

1.  **Allocation:** Use `malloc`, `calloc`, or custom wrappers like `aligned_alloc_64` (in `tensor.h`).
2.  **Deallocation:** Every allocation must have a matching `free` or custom deallocation function (`free_arr`, `free_tensor`).
3.  **Graph Tensors:** Tensors created during the forward pass (e.g., via `matmul`, `relu`) must be explicitly freed after `backward()` to avoid memory leaks. See `train.c` for examples.
4.  **Error Handling on Allocation:** Always check for `NULL` return from `malloc`/`calloc` and handle it gracefully (usually by calling `perror` and `exit(1)`).

### 3.4. Imports and Macros

- **Headers:** Use `#include "..."` for local headers and `#include <...>` for system/library headers.
- **Macro Guards:** Use `#pragma once` in headers where possible (e.g., `tensor.h`).
- **Function Style:** Prefer `static inline` for utility and core tensor operations in the header file (`tensor.h`) to promote optimization and header-only style.

### 3.5. Error Handling

- **Critical Errors:** Use `perror()` or `fprintf(stderr, ...)` to print a descriptive error message to standard error, followed by an immediate program termination via `exit(1)` or `return 1` in `main`.
- **No Exceptions:** Pure C means no C++ exceptions. Use return values for non-critical status if necessary, but critical failures should halt the program immediately.

## 4. Performance Guidelines

- **Alignment:** Prioritize 64-byte alignment for performance-critical tensor data (handled by `aligned_alloc_64`). When creating new tensor data structures, ensure they follow this pattern.
- **Platform Optimization:** Be mindful of the `USE_ACCELERATE` macro. Any new tensor operations should check for the presence of Accelerate on macOS and fall back to optimized OpenMP or standard C loops.
