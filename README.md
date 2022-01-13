# FACT AO

Read the README before contributing to this project

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

1. Install the necessary software to use:

- python 3.8.7
  https://www.python.org/downloads/release/python-387/
  check version
  ```sh
    python --version
  ```

### Installation

1. Clone the repo
   ```sh
   git clone git@github.com:yrunhaar/fact-ai.git
   ```
   And visit the created directory
   ```sh
   cd .\fact-ai\
   ```
2. Create a virtual environment
   ```sh
   python -m venv env
   ```
   And activate the virtual environment
   ```sh
   .\env\Scripts\activate
   ```
3. Go to DECAF folder
    ```sh
   cd DECAF
    ```
4. Install Python packages
   ```sh
   pip install -r requirements.txt
   pip install .
   ```

## Run tests

1. Make sure you are in the virtual environment (env). Otherwise, activate the virtual environment
   ```sh
   .\env\Scripts\activate
   ```
2. Go to DECAF folder
    ```sh
   cd DECAF
    ```
4. Install Python packages
   ```sh
    pip install -r requirements_dev.txt
    pip install .
   ```
5. Run the application
    ```sh
    pytest -vsx
   ```

## Run examples

1. Make sure you are in the virtual environment (env). Otherwise, activate the virtual environment
   ```sh
   .\env\Scripts\activate
   ```
2. Go to DECAF folder
    ```sh
   cd DECAF
    ```
4. Install Python packages
   ```sh
    pip install -r requirements_dev.txt
    pip install .
   ```
5. Run example: Base example on toy dag
    ```sh
    cd tests
    python run_example.py
   ```
6. Run example: Example to run with a dataset size of 2000 for 300 epochs
    ```sh
    python run_example.py --datasize 2000 --epochs 300
   ```
