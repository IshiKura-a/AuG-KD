<div align='center'>

<h2><a href="https://openreview.net/forum?id=fcqWJ8JgMR">[ICLR'24 Poster] AuG-KD: Anchor-Based Mixup Generation for Out-of-Domain Knowledge Distillation</a></h2>

[Zihao Tang](https://github.com/IshiKura-a/)<sup>1</sup>, [Zheqi Lv](https://github.com/HelloZicky)<sup>1</sup>, [Shengyu Zhang](https://shengyuzhang.github.io/)<sup>1*</sup>, [Yifan Zhou](https://github.com/Geniusyingmanji)<sup>2</sup>, Xinyu Duan<sup>3</sup>, [Fei Wu](https://mypage.zju.edu.cn/wufei)<sup>1</sup>, [Kun Kuang](https://kunkuang.github.io/)<sup>1*</sup>
 
<sup>1</sup>[ZJU](https://www.zju.edu.cn/english/), <sup>2</sup>[SJTU](https://en.sjtu.edu.cn/), <sup>3</sup>Huawei Cloud
<br> <sup>*</sup>Corresponding Authors
</div>

Official Pytorch Implementation for the research paper titled "AuG-KD: Anchor-Based Mixup Generation for Out-of-Domain Knowledge Distillation".

## Installation
Clone this repository and install the required packages:
```shell
git clone https://github.com/IshiKura-a/AuG-KD.git
cd AuG-KD

conda create -n AuG-KD python=3.7
conda activate AuG-KD
conda install pytorch torchvision torchaudio pytorch-cuda=12.0 -c pytorch -c nvidia

pip install -r requirements.txt
```
Download datasets:
* [Office-31](https://www.cc.gatech.edu/~judy/domainadapt/)
* [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html)
* [VisDA-2017](https://ai.bu.edu/visda-2017/)

## Train

For example, to get similar results in Office (Amazon, Webcam → DSLR), you should first train the teacher model with the code below:run this code
```shell
model=resnet_34
root_dir=xxx
CUDA_VISIBLE_DEVICES=0 python main.py \
  --batch_size 2048 \
  --teacher "${model}" \
  --dataset office \
  --log_dir ${root_dir}/model/"${model}"/office_AW \
  --seed 2023 \
  train_teacher \
  --t_lr 1e-3 \
  --t_wd 1e-4 \
  --t_epoch 400 \
  --t_mode auto_split \
  --t_data ${root_dir}/dataset/office/amazon/images ${root_dir}/dataset/office/webcam/images
```
Then, use teacher model to train its OOD student:
```shell
root_dir=xxx
teacher=resnet34
student=mobilenet_v3_small
CUDA_VISIBLE_DEVICES=0 python main.py \
    --teacher ${teacher} \
    --teacher_dir ${root_dir}/model/${teacher}/office_AW/${teacher}_ckpt.pt \
    --student ${student} \
    --latent_size 100 \
    --dataset office \
    --target ${root_dir}/dataset/office/dslr/images \
    --test ${root_dir}/dataset/office/dslr/images \
    --lr 1e-3 \
    --wd 1e-4 \
    --epoch 200 \
    --batch_size 2048 \
    --seed ${seed} \
    --d_lr 1e-3 \
    --e_lr 1e-4 \
    --g_epoch 200 \
    --a_lr 1e-3 \
    --a_epoch 200 \
    --invariant 0.25 \
    --log_dir ${root_dir}/model/GenericKD/ \
    --postfix s${seed}_AW \
    --a 0.6 \
    --b 0.2
```
For other settings, we refer readers to Appendix B for hyperparameter settings to replicate our results.
## Acknowledgement
This work was supported by National Key R\&D Program of China (No. 2022ZD0119100), the National Natural Science Foundation of China (62376243, 62037001, U20A20387), Scientific Research Fund of Zhejiang Provincial Education Department (Y202353679), and the StarryNight Science Fund of Zhejiang University Shanghai Institute for Advanced Study (SN-ZJU-SIAS-0010).

## Citation
```bib
@inproceedings{
tang2024augkd,
title={AuG-{KD}: Anchor-Based Mixup Generation for Out-of-Domain Knowledge Distillation},
author={Zihao TANG and Shengyu Zhang and Zheqi Lv and Yifan Zhou and Xinyu Duan and Kun Kuang and Fei Wu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=fcqWJ8JgMR}
}
```
