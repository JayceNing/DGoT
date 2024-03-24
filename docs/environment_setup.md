# Environment Setup

In this tutorial, we will explain how to configure the necessary environment for DGoT manually. For Docker setup, please refer to the [Quick Start](https://github.com/JayceNing/DGoT?tab=readme-ov-file#quick-start) tutorial.

Recommended Environment:
* Python >= 3.8
* Pytorch >= 2.0.0
* Transformers >= 4.34 (Different open-source large language models may have different requirements for Transformers versions. Here, InternLM2 is used as an example.)

GPU requirement:
* GeForce RTX 3090 or above (our experiments are conducted on 3090).


```bash
# Creating Conda Environment
conda create -n dgot python=3.8
# Activating Conda Environment
conda activate dgot
```

## Install PYROUGE

Through testing, we need to clone these two repositories to complete the PYROUGE installation.

* [https://github.com/andersjo/pyrouge](https://github.com/andersjo/pyrouge)
* [https://github.com/bheinzerling/pyrouge](https://github.com/bheinzerling/pyrouge)


Set up the folder and clone the repository
```bash
# Use the setup.py file from this repository
mkdir install_pyrouge
cd install_pyrouge
git clone https://github.com/andersjo/pyrouge.git

# Download ROUGE-1.5.5 through this repository
cd ..
git clone https://github.com/bheinzerling/pyrouge
```

Install PYROUGE
```bash
cd install_pyrouge/pyrouge
python setup.py install

# The path needs to be set to the absolute path of the ROUGE-1.5.5 folder downloaded from the https://github.com/bheinzerling/pyrouge repository.
pyrouge_set_rouge_path /PATH/TO/pyrouge/tools/ROUGE-1.5.5/
```

Tests whether the installation is successful.
```bash
python -m pyrouge.test
```