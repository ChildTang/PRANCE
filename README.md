

# PRANCE：Joint Token-Optimization and Structural Channel-Pruning for Adaptive ViT Inference

This is the official implementation of PRANCE.

**Abstract:** We introduce PRANCE, a Vision Transformer compression framework that jointly optimizes the activated channels and reduces tokens, based on the characteristics of inputs. Specifically, PRANCE leverages adaptive token optimization strategies for a certain computational budget, aiming to accelerate ViTs' inference from a unified data and architectural perspective. However, the joint framework poses challenges to both architectural and decision-making aspects. Firstly, while ViTs inherently support variable-token inference, they do not facilitate dynamic computations for variable channels. To overcome this limitation, we propose a meta-network using weight-sharing techniques to support arbitrary channels of the Multi-head Self-Attention and Multi-layer Perceptron layers, serving as a foundational model for architectural decision-making. Second, simultaneously optimizing the structure of the meta-network and input data constitutes a combinatorial optimization problem with an extremely large decision space, reaching up to around 1014, making supervised learning infeasible. To this end, we design a lightweight selector employing Proximal Policy Optimization for efficient decision-making. Furthermore, we introduce a novel "Result-to-Go" training mechanism that models ViTs' inference process as a Markov decision process, significantly reducing action space and mitigating delayed-reward issues during training. Extensive experiments demonstrate the effectiveness of PRANCE in reducing FLOPs by approximately 50%, retaining only about 10% of tokens while achieving lossless Top-1 accuracy. Additionally, our framework is shown to be compatible with various token optimization techniques such as pruning, merging, and sequential pruning-merging strategies.

## 1. Cloning the Repository

```
git clone https://github.com/ChildTang/PRANCE.git
```

## 2. Setup

You can use the following command to setup the training/evaluation environment:

```
conda create -n prance python=3.9
conda activate prance
pip install -r requirements.txt
```

We use the ImageNet dataset at http://www.image-net.org/. The training set is moved to /path_to_imagenet/imagenet/train and the validation set is moved to /path_to_imagenet/imagenet/val:

```
/path_to_imagenet/imagenet/
  train/
    class0/
      img0.jpeg
      ...
    class1/
      img0.jpeg
      ...
    ...
  val/
    class0/
      img0.jpeg
      ...
    class1/
      img0.jpeg
      ...
    ...
```

The checkpiont of meta-network and PPO selector can be download at:

```
https://huggingface.co/childtang/PRANCE/tree/main
```

The structure is as follows:

```
checkpoints/
    finetune_meta-network/
        merge/
            small/
                acc_80_flops_2.8_param_24_tk_0.38/
                    checkpoint.pth
                ...
            tiny/
                ...
            base/
								...
        prune/
            ...
        prune-then-merge/
            ...
    meta-network/
        small/
            checkpoint.pth
        tiny/
        base/
    ppo/
        merge/
            small/
                acc_80_flops_2.8_param_24_tk_0.38/
                    actor.pth
                    critic.pth
                ...
            tiny/
            base/
        prune/
        prune-then-merge/
```

## 3. Usage

### Test

You can test the model by run the test.sh, and change the settings as you need:

```
sh test.sh
```

## 4. Acknowledgement

The authors would like to thank the following insightful open-source projects & papers, this work cannot be done without all of them:

1. AutoFormer: https://github.com/microsoft/Cream/tree/main/AutoFormer
2. ToMe: https://github.com/facebookresearch/ToMe
3. Diffrate: https://github.com/OpenGVLab/DiffRate 

## Citation

If our work contribute to your work, you can cite our work as follows:

```
@article{li2024prance,
  title={PRANCE: Joint Token-Optimization and Structural Channel-Pruning for Adaptive ViT Inference},
  author={Li, Ye and Tang, Chen and Meng, Yuan and Fan, Jiajun and Chai, Zenghao and Ma, Xinzhu and Wang, Zhi and Zhu, Wenwu},
  journal={arXiv preprint arXiv:2407.05010},
  year={2024}
}
```

