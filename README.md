# Expediting Large-Scale Vision Transformer for Dense Prediction without Fine-tuning

## Introduction

This is the official implementation of the paper "[Expediting Large-Scale Vision Transformer for Dense Prediction without Fine-tuning](https://arxiv.org/abs/2210.01035)" on [OneFormer](https://arxiv.org/abs/2211.06220). 

![framework](images/Hourglass_swin_framework.png)
![framework](images/TokenClusterReconstruct_Details.png)


## Results 

Here we implement our method on Swin backbone. Thus we report the GFLOPs and FPS of backbone. 

### ADE20K

| Method           | Backbone | $\alpha$ | h $\times$ w   | GFLOPs | FPS   | PQ  | AP | mIoU |
| ---------------- | -------- | -------- | -------------- | ------ | ----- | ----- | ----- | ----- |
| OneFormer  | Swin-L | -        | 12 $\times$ 12 | 1206   | 3.52 | 51.3 | 37.7 | 56.9 |
| OneFormer + Ours | Swin-L | 8       |  10 $\times$ 10  | 1029   | 4.02 | 51.1 | 36.8 | 57 |
| OneFormer + Ours | Swin-L | 10      |  8 $\times$ 8  | 898    | 4.58 | 50.7 | 36.7 | 56.5 |
| OneFormer + Ours | Swin-L | 8      |  8 $\times$ 8  | 846    | 4.85 | 50.5 | 36.4 | 55.9 |

## Installation Instructions

- We use Python 3.8, PyTorch 1.10.1 (CUDA 11.3 build).
- We use Detectron2-v0.6.
- For complete installation instructions, please see [INSTALL.md](INSTALL.md).

## Dataset Preparation

- We experiment on ADE20K benchmark. You can try our method on other benchmark such as Cityscapes and COCO 2017.
- Please see [Preparing Datasets for OneFormer](datasets/README.md) for complete instructions for preparing the datasets.

## Evaluation Instructions

- You need to pass the value of `task` token. `task` belongs to [panoptic, semantic, instance].

```bash
python train_net.py --dist-url 'tcp://127.0.0.1:50164' \
    --num-gpus 8 \
    --config-file configs/ade20k/swin/oneformer_hourglass_swin_large_bs16_160k_1280x1280.yaml \
    --eval-only MODEL.IS_TRAIN False MODEL.WEIGHTS <path-to-checkpoint> \
    MODEL.TEST.TASK <task>
```

## License

This project is released under the [Apache 2.0 license](LICENSE).


## Acknowledgement
The repo is built based on [OneFormer](https://github.com/SHI-Labs/OneFormer). We thank the authors for their great work.

## Citation
If you find this project useful in your research, please consider cite:

```BibTex
@article{liang2022expediting,
	author    = {Liang, Weicong and Yuan, Yuhui and Ding, Henghui and Luo, Xiao and Lin, Weihong and Jia, Ding and Zhang, Zheng and Zhang, Chao and Hu, Han},
	title     = {Expediting large-scale vision transformer for dense prediction without fine-tuning},
	journal   = {arXiv preprint arXiv:2210.01035},
	year      = {2022},
}
```

```BibTex
@inproceedings{jain2023oneformer,
      title={{OneFormer: One Transformer to Rule Universal Image Segmentation}},
      author={Jitesh Jain and Jiachen Li and MangTik Chiu and Ali Hassani and Nikita Orlov and Humphrey Shi},
      journal={CVPR}, 
      year={2023}
    }
```
