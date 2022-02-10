# FACT AI

Read the README before contributing to this project

## Getting Started

To get a local copy up and reproduce our experiments follow these simple steps.

### Prerequisites

1. Install the necessary software to use:

- conda:
  https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

### Installation

1. Clone the repo
2. Create a conda environment
   ```sh
   conda env create --file environment.yml
   ```

## Run notebook

1. Activate conda environment (see installation)
   ```sh
   conda activate factai
   ```
2. Start jupyter notebook environment
   ```sh
   jupyter notebook
   ```
3. Open decaf_reproducibility.ipynb
4. Under Kernel select Restart & Run all

## Run FairGan

Due to dependency conflicts between the FairGAN code using tensorflow 1.15.4 (tensorboard<1.16.0 and >=1.15.0) and the Decaf code using pytorch-lightning 1.4.9 (tensorboard>=2.2.0), the FairGAN results have to generated in a separate environment.

1. Install the necessary software to use:
   - Java 8 or higher (for javabridge):
     https://pythonhosted.org/javabridge/installation.html
2. Create a conda environment
   ```sh
   conda env create --file environment_fairgan.yml
   ```
   - if error when installing javabridge: https://www.tutorialexample.com/fix-python-pip-install-link-fatal-error-lnk1158-cannot-run-rc-exe-error-python-tutorial/
3. Activate conda environment (see installation)
   ```sh
   conda activate fairgan
   ```
4. Start jupyter notebook environment
   ```sh
   jupyter notebook
   ```
5. Open fairgan_reproducibility.ipynb
6. Under Kernel select Restart & Run all
