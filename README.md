# DGoT: Dynamic Graph of Thoughts for Scientific Abstract Generation

![](./paper/img/overall_process.png)

This is the implementation of our LREC-COLING 2024 paper [DGoT: Dynamic Graph of Thoughts for Scientific Abstract Generation](https://arxiv.org/abs/2403.17491).

## Quick Start

### Environment Setup

We strongly recommend that you use Docker images to run our programs.

```bash
# pull docker image
docker pull jaycening/dgot_demo:v1.0.0

# Clone this repository and mount it into the image container.
git clone https://github.com/JayceNing/DGoT.git
docker run --gpus all -it -d --privileged=true -v ./DGoT:/home/nxy/LLM/DGoT

# Enter the target folder.
docker exec -it dgot /bin/bash
cd /home/nxy/LLM/DGoT
```

The image is preconfigured with InternLM2 deployed under LMDeploy version 0.2.4.

If you want to configure the environment manually, see the documentation [environment_setup.md](https://github.com/JayceNing/DGoT/blob/master/docs/environment_setup.md)

### Data Preparation for PubMedCite Dataset

The citation graph records of PubMedCite come from the [CitationSum repository](https://github.com/zhehengluoK/CitationSum). Here, we provide code for downloading datasets based on the official [PubMed](https://pubmed.ncbi.nlm.nih.gov/download/) API.

```bash
python get_data.py --required_num 100
```

* `required_num` is the number of data entries required for the training and testing datasets to be downloaded.

### Activate LLM's API service

Taking InternLM2 as an example.
```bash
cd /home/nxy/internlm2_chat_deploy
lmdeploy serve api_server ./workspace --cache-max-entry-count 0.2
```

### Training Process
```bash
cd /home/nxy/LLM/DGoT
python generate_abstract.py --begin 0 --end 1 --mode "train" --model "internlm2" --task "default"
```

* `begin` and `end` indicate the starting and ending indices of the dataset being used.
* `mode` indicates whether the dataset being used is train dataset or test dataset.
* `model` represents the LLM used for inference.
* `task` represents the type of task being performed.

### Reasoning Process
```bash
python generate_abstract.py --begin 0 --end 100 --mode "test" --model "internlm2" --task "default" --thresh_g 0.34 --thresh_a 0.35 --thresh_i 0.34
```

* `thresh_g`, `thresh_a` and `thresh_i` respectively represent the thresholds used for generating transformation, aggregating transformation, and boosting transformation in DGoT.

## Acknowledgement

This work is based on the following prompt framework, large language model, and model deployment toolkit. Thanks for the open source contribution!

* [Graph of Thoughts (GoT)](https://github.com/spcl/graph-of-thoughts)
* [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)
* [InternLM](https://github.com/InternLM/InternLM)
* [LMDeploy](https://github.com/InternLM/lmdeploy)

This paper's computational resources are supported by the High-performance Computing Platform of BUPT.

## Citations

If you find this repository valuable, please give it a star!

Using this in your work? Please reference us using the provided citation:


```
@misc{ning2024dgot,
      title={DGoT: Dynamic Graph of Thoughts for Scientific Abstract Generation}, 
      author={Xinyu Ning and Yutong Zhao and Yitong Liu and Hongwen Yang},
      year={2024},
      eprint={2403.17491},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contributor

<a href="https://github.com/JayceNing/ChatBrain/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=JayceNing/ChatBrain" />
</a>

Jayce Ning

Home Page：https://jaycening.github.io/zh-cn/

Github：https://github.com/JayceNing

ZhiHu：https://www.zhihu.com/people/XinyuNing

