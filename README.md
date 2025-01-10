

# PRANCE：Joint Token-Optimization and Structural Channel-Pruning for Adaptive ViT Inference

This is the official implementation of PRANCE.

**Abstract:** We introduce PRANCE, a Vision Transformer compression framework that jointly optimizes the activated channels (dynamic channel pruning) and reduces tokens (dynamic token pruning (or) merging (or) pruning-then-merging), based on the characteristics of inputs. PRANCE leverages adaptive token optimization strategies for a certain computational budget, aiming to accelerate ViTs' inference from a unified data and architectural perspective. PRANCE can reduces FLOPs by approximately 50%, retaining only about 10% of tokens while achieving lossless Top-1 accuracy. 

## 1. Cloning the Repository

```
git clone https://github.com/ChildTang/PRANCE.git
```

## 2. Setup

- You can use the following command to setup the training/evaluation environment:  

```
conda create -n prance python=3.9
conda activate prance
pip install -r requirements.txt
```

- Install the NVIDIA-dali framework (for training): 
https://github.com/NVIDIA/DALI?tab=readme-ov-file#installing-dali



## 3. Data
- We use the ImageNet dataset at http://www.image-net.org/. The training set is moved to /path_to_imagenet/imagenet/train and the validation set is moved to /path_to_imagenet/imagenet/val:

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

## 4. Pretrained Weights
Download the pretrained checkpionts of meta-network and PPO selector:

https://huggingface.co/childtang/PRANCE/tree/main

The overall structure of the project is as follows:

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

## 5. Usage
### Training

This part will be released soon. 

### Finetuning

This part will be released soon.

### Testing

You can test the model by running the test.sh, and change the settings in the test.sh file as you need:

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

