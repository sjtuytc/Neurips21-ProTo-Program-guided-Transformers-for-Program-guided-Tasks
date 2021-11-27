# ProTo: Program-Guided Transformer for Program-Guided Tasks

This repository is the official implementation of [ProTo: Program-Guided Transformer for Program-Guided Tasks](https://arxiv.org/abs/2110.00804). 

Youtube link: https://www.youtube.com/watch?v=Dd8EHqMfOPI&t=24s

Exemplar program-guided tasks:

![image-20211103105216886](images/task_description.png)



ProTo model illustrated:

![image-20211103103539274](images/proto_model.png)

## Abstract:

Programs, consisting of semantic and structural information, play an important role in the communication between humans and agents. Towards learning general program executors to unify perception, reasoning, and decision making, we formulate program-guided tasks which require learning to execute a given program on the observed task specification. Furthermore, we propose the Program-guided Transformer (ProTo), which integrates both semantic and structural guidance of a program by leveraging cross-attention and masked self-attention to pass messages between the specification and routines in the program. ProTo executes a program in a learned latent space and enjoys stronger representation ability than previous neural-symbolic approaches. We demonstrate that ProTo significantly outperforms the previous state-of-the-art methods on GQA visual reasoning and 2D Minecraft policy learning datasets. Additionally, ProTo demonstrates better generalization to unseen, complex, and human-written programs.

## News

**2021.Nov.27**  Code is available at Github. Doc is constructing.

## Requirements

To install requirements:

```shell
# create conda environment
conda create --name proto python=3.7
conda activate proto
# install packages
pip install -r requirements.txt
```

#### GQA

1. Feature abstraction via [bottom up attention](https://github.com/MILVLG/bottom-up-attention.pytorch#Pre-trained-models):

```shell
# clone repo
git clone --recursive https://github.com/MILVLG/bottom-up-attention.pytorch
# install detectron2
cd detectron2
pip install -e .
cd ..
# install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install
cd ..
# install bottom up attention repo
python setup.py build develop
pip install ray
```

#### Minecraft

The original 

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

#### GQA

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

#### Minecraft

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

## Reference and acknowledgement

Cite our paper:

@misc{zhao2021proto,
      title={ProTo: Program-Guided Transformer for Program-Guided Tasks}, 
      author={Zelin Zhao and Karan Samel and Binghong Chen and Le Song},
      year={2021},
      eprint={2110.00804},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

## Contributing

Please raise an issue if you found a bug and we welcome pull requests.
