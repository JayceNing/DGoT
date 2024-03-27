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

## Dependencies
First, clone this repository.
```bash
git clone https://github.com/JayceNing/DGoT.git
cd DGoT
```

Then, install dependencies using pip.

```bash
pip install -r requirements.txt
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
git clone https://github.com/bheinzerling/pyrouge.git

# Download ROUGE-1.5.5 through this repository
cd ..
git clone https://github.com/andersjo/pyrouge.git
```

Install PYROUGE
```bash
cd install_pyrouge/pyrouge
python setup.py install

# The path needs to be set to the absolute path of the ROUGE-1.5.5 folder 
# downloaded from the https://github.com/bheinzerling/pyrouge repository.
pyrouge_set_rouge_path /PATH/TO/pyrouge/tools/ROUGE-1.5.5/
```

Tests whether the installation is successful.
```bash
python -m pyrouge.test
```

If the following output is displayed, the installation is correct. 

```
----------------------------------------------------------------------
Ran 10 tests in 3.822s

OK
```

Otherwise, check the [FAQ](https://github.com/JayceNing/DGoT/blob/master/docs/environment_setup.md#FAQ) at the bottom.

## LLM Deploy
For deployment of ChatGLM-6B and InternLM2-Chat-7B, please refer to the respective repositories (We accelerated the deployment of InternLM using LMDeploy.).

* [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)
* [InternLM](https://github.com/InternLM/InternLM)
* [LMDeploy](https://github.com/InternLM/lmdeploy)

If you want to test the performance of DGoT based on InterLM, we strongly recommend using our image environment [dgot_demo](https://hub.docker.com/r/jaycening/dgot_demo/tags).

## FAQ
### PYROUGE installation issue

#### db file error
```
Cannot open exception db file for reading: /PATH/TO/pyrouge/tools/ROUGE-1.5.5/data/WordNet-2.0.exc.db
````
Solution
```bash
cd /PATH/TO/pyrouge/tools/ROUGE-1.5.5/data/WordNet-2.0-Exceptions/;
./buildExeptionDB.pl . exc WordNet-2.0.exc.db;
cd ../;
rm WordNet-2.0.exc.db;
ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db;
````

**Note**: Please replace `/PATH/TO/` with the actual location of your `pyrouge/tools/ROUGE-1.5.5/`.