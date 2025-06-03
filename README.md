# simple-code-execution

[![Documentation Status](https://github.com/gabeorlanski/simple-code-execution/workflows/Build%20and%20Deploy%20Documentation/badge.svg)](https://gabeorlanski.github.io/simple-code-execution/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A powerful Python library for executing code predictions through subprocess and threading with comprehensive parallel processing, file management, and result handling.

## 🚀 Features

- **⚡ Parallel Execution**: Execute multiple code predictions simultaneously using multiprocessing
- **📁 Automatic File Management**: Write, execute, and cleanup temporary files seamlessly
- **🛡️ Robust Error Handling**: Built-in timeout handling, syntax error detection, and graceful failure recovery
- **⚙️ Flexible Configuration**: Comprehensive configuration options for execution behavior
- **🔄 Processing Pipeline**: Powerful preprocessing and postprocessing pipeline for custom workflows
- **📊 Resource Monitoring**: Memory and CPU usage monitoring with configurable limits
- **🎯 Production Ready**: Battle-tested for large-scale code execution workloads

## 📦 Installation

```bash
pip install simple-code-execution
```

### Requirements

- Python 3.10+
- psutil >= 5.9
- numpy >= 1.26
- aiofiles >= 22.1.0
- tqdm >= 4.60.0
- ujson >= 5.10.0

## 🔥 Quick Start

```python
from code_execution import ExecutionConfig, execute_predictions, Executable, Command

# Define your code predictions
predictions = [
    {"id": 1, "code": "print('Hello, World!')"},
    {"id": 2, "code": "x = 5\nprint(x * 2)"},
    {"id": 3, "code": "import math\nprint(math.sqrt(16))"},
]

# Configure execution settings
config = ExecutionConfig(
    num_workers=2,           # Number of parallel workers
    default_timeout=10,      # Timeout in seconds
    max_execute_at_once=3    # Max concurrent executions
)

# Define preprocessor: converts predictions to executable commands
def preprocessor(prediction):
    return Executable(
        files={"main.py": prediction["code"]},  # Files to write
        commands=[Command(command=["python3", "main.py"])],  # Commands to run
        tracked_files=[]  # Files to read back after execution
    )

# Define postprocessor: processes execution results
def postprocessor(prediction, result):
    return {
        "id": prediction["id"],
        "code": prediction["code"],
        "output": result.command_results[0].stdout,
        "success": result.command_results[0].return_code == 0,
        "runtime": result.command_results[0].runtime
    }

# Execute all predictions
results = execute_predictions(
    config=config,
    pred_list=predictions,
    preprocessor=preprocessor,
    postprocessor=postprocessor
)

# Print results
for result in results.results:
    print(f"ID: {result['id']}")
    print(f"Output: {result['output'].strip()}")
    print(f"Success: {result['success']}")
    print(f"Runtime: {result['runtime']:.3f}s")
    print("-" * 40)
```

**Output:**

```
ID: 1
Output: Hello, World!
Success: True
Runtime: 0.045s
----------------------------------------
ID: 2
Output: 10
Success: True
Runtime: 0.043s
----------------------------------------
ID: 3
Output: 4.0
Success: True
Runtime: 0.051s
----------------------------------------
```

## 🏗️ Architecture

The library follows a simple but powerful workflow:

1. **Preprocess** → Convert your data into `Executable` objects
2. **Execute** → Run code in parallel with resource management
3. **Postprocess** → Combine results with original predictions

```mermaid
graph LR
    A[Predictions] --> B[Preprocessor]
    B --> C[Executor]
    C --> D[Postprocessor]
    D --> E[Results]
```

## ⚙️ Configuration

```python
config = ExecutionConfig(
    num_workers=4,              # Parallel workers
    default_timeout=30,         # Default timeout per command
    max_execute_at_once=10,     # Max concurrent executions
    write_rate_limit=768,       # File writing rate limit
    display_write_progress=True # Show progress bars
)
```

## 🎯 Use Cases

- **Code Generation Evaluation**: Test AI-generated code at scale
- **Competitive Programming**: Run solutions against test cases
- **Code Analysis**: Execute and analyze code behavior
- **Educational Tools**: Safe code execution in learning environments
- **Research**: Large-scale code execution experiments

## ⚡ Advanced Features

### Multiple Commands per Prediction

```python
def multi_command_preprocessor(prediction):
    return Executable(
        files={
            "setup.py": "# Setup code",
            "main.py": prediction["code"]
        },
        commands=[
            Command(command=["python3", "setup.py"]),
            Command(command=["python3", "main.py"], timeout=5)
        ],
        tracked_files=["output.txt"]  # Read this file after execution
    )
```

### Custom Early Stopping

```python
def custom_early_stop(cmd_idx, result):
    # Stop if command fails
    if result.return_code != 0:
        return True
    # Stop if output contains error
    if "error" in result.stdout.lower():
        return True
    return False

executable = Executable(
    files={"test.py": code},
    commands=[Command(command=["python3", "test.py"])],
    should_early_stop=custom_early_stop
)
```

## ⚠️ Important Notes

### Pickleable Functions Required

**Both preprocessor and postprocessor functions must be pickleable** (serializable) for multiprocessing:

✅ **Good:**

```python
def my_preprocessor(prediction):
    return Executable(...)
```

❌ **Bad:**

```python
# Lambda - not pickleable
preprocessor = lambda pred: Executable(...)

# Nested function - not pickleable
def outer():
    def preprocessor(pred):
        return Executable(...)
    return preprocessor
```

## 📚 Documentation

- **[Quick Start Guide](https://gabeorlanski.github.io/simple-code-execution/quickstart.html)** - Get up and running in minutes
- **[API Reference](https://gabeorlanski.github.io/simple-code-execution/api/modules.html)** - Complete API documentation
- **[Full Documentation](https://gabeorlanski.github.io/simple-code-execution/)** - Comprehensive guides and examples

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
git clone https://github.com/gabeorlanski/simple-code-execution.git
cd simple-code-execution
pip install -e .
pip install -r docs/requirements.txt

# Run tests
pytest

# Build documentation locally
cd docs
make html
make serve  # Serves at http://localhost:8000
```

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for reliable, scalable code execution in research and production environments
- Designed with safety and resource management as core principles
- Optimized for both single-use scripts and long-running services

---

**Made with ❤️ by [Gabriel Orlanski](https://github.com/gabeorlanski)**
