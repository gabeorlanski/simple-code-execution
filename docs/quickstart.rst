Quick Start Guide
==================

The **simple-code-execution** library provides a powerful framework for executing code predictions in parallel with comprehensive logging, file management, and result processing.

Installation
------------

.. code-block:: bash

    pip install simple-code-execution

Basic Usage
-----------

The library follows a simple workflow: **preprocess** → **execute** → **postprocess**. Here's a minimal example that demonstrates executing Python code snippets:

.. code-block:: python

    from code_execution import ExecutionConfig, execute_predictions, Executable, Command

    # Define your predictions (code to execute)
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
    # IMPORTANT: Must be pickleable (serializable) for multiprocessing
    def preprocessor(prediction):
        return Executable(
            files={"main.py": prediction["code"]},  # Files to write
            commands=[Command(command=["python3", "main.py"])],  # Commands to run
            tracked_files=[]  # Files to read back after execution
        )

    # Define postprocessor: processes execution results
    # IMPORTANT: Must be pickleable (serializable) for multiprocessing
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

Expected Output:

.. code-block:: text

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

Key Components
--------------

**ExecutionConfig**
    Configuration object that controls execution behavior including parallelism, timeouts, and resource limits.

**Executable**
    Data structure containing files to write, commands to execute, and files to track after execution.

**Command**
    Represents a single command with its arguments, timeout, and execution parameters.

**Preprocessor Function**
    Converts your prediction data into ``Executable`` objects that the library can run. **Must be pickleable** (serializable) as it runs in separate processes.

**Postprocessor Function**
    Processes execution results and combines them with original predictions for final output. **Must be pickleable** (serializable) as it runs in separate processes.

.. important::
   **Pickleable Functions Required**
   
   Both preprocessor and postprocessor functions must be **pickleable** (serializable with Python's ``pickle`` module) because the library uses multiprocessing for parallel execution. This means:
   
   - Use regular functions, not lambdas or nested functions
   - Avoid closures that capture local variables
   - Don't use instance methods from complex objects
   - All imported modules must be available in worker processes
   
   **Good:**
   
   .. code-block:: python
   
       def my_preprocessor(prediction):
           return Executable(...)
   
   **Bad:**
   
   .. code-block:: python
   
       # Lambda - not pickleable
       preprocessor = lambda pred: Executable(...)
       
       # Nested function - not pickleable
       def outer():
           def preprocessor(pred):
               return Executable(...)
           return preprocessor

Advanced Features
-----------------

**Multiple Commands per Prediction**

.. code-block:: python

    def advanced_preprocessor(prediction):
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

**Error Handling**

The library automatically handles timeouts, syntax errors, and runtime failures. Check ``result.had_error`` and ``result.timed_out`` properties for execution status.

**Parallel Execution Control**

Use ``max_execute_at_once`` to limit concurrent executions and prevent resource exhaustion:

.. code-block:: python

    config = ExecutionConfig(
        num_workers=4,
        max_execute_at_once=10,  # Execute max 10 predictions simultaneously
        default_timeout=30
    )

Next Steps
----------

- Explore the :doc:`api_reference` for complete API documentation
- Check out :doc:`advanced_usage` for complex execution scenarios
- See :doc:`configuration` for all available configuration options
