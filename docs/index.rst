simple-code-execution Documentation
====================================

**simple-code-execution** is a Python library for executing code predictions through subprocess and threading with comprehensive parallel processing, file management, and result handling.

Features
--------

- **Parallel Execution**: Execute multiple code predictions simultaneously using multiprocessing
- **File Management**: Automatic writing and cleanup of temporary files
- **Robust Error Handling**: Built-in timeout handling, syntax error detection, and graceful failure recovery
- **Flexible Configuration**: Comprehensive configuration options for execution behavior
- **Result Processing**: Powerful preprocessing and postprocessing pipeline for custom workflows
- **Resource Control**: Memory and CPU usage monitoring with configurable limits

Quick Example
-------------

.. code-block:: python

    from code_execution import ExecutionConfig, execute_predictions, Executable, Command

    # Configure execution
    config = ExecutionConfig(num_workers=2, default_timeout=10)

    # Define predictions
    predictions = [{"code": "print('Hello, World!')"}]

    # Execute with custom preprocessor
    def preprocessor(pred):
        return Executable(
            files={"main.py": pred["code"]},
            commands=[Command(command=["python3", "main.py"])]
        )

    results = execute_predictions(config, predictions, preprocessor)

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/modules

Installation
------------

Install using pip:

.. code-block:: bash

    pip install simple-code-execution

Or install from source:

.. code-block:: bash

    git clone https://github.com/gabeorlanski/simple-code-execution.git
    cd simple-code-execution
    pip install -e .

Requirements
~~~~~~~~~~~~

- Python 3.10+
- psutil >= 5.9
- numpy >= 1.26
- aiofiles >= 22.1.0
- tqdm >= 4.60.0
- ujson >= 5.10.0

Contributing
------------

Contributions are welcome! Please feel free to submit a Pull Request.

License
-------

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 