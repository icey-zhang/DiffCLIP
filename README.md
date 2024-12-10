<div align="center">
<h1> DiffCLIP </h1>
<h3> DiffCLIP: Few-shot Language-driven Multimodal Classifier </h3>
<h4> AAAI 2025</h4>
  
</div>


## **Overview**

<p align="center">
  <img src="assets\DiffCLIP.png" alt="overview" width="90%">
</p>

## **Getting Started**

**Step 1: Clone the DiffCLIP repository:**

To get started, first clone the DiffCLIP repository and navigate to the project directory:

```bash
git clone https://github.com/icey-zhang/DiffCLIP
cd DiffCLIP
```

**Step 2: Environment Setup:**

DiffCLIP recommends setting up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:

***Create and activate a new conda environment***

```bash
conda create -n DiffCLIP python=3.9.17
conda activate DiffCLIP
```

***install some necessary package***
```bash
pip install pytorch
......
```

### Prepare the dataset

```python
root
├── Trento
│   ├── HSI.mat
│   ├── LiDAR.mat
│   ├── TRLabel.mat
│   ├── TSLabel.mat
├── ......

```

### Begin to train

```python
python train.py
```

## Citation
If our code is helpful to you, please cite:

```
@article{zhang2024multimodal,
  title={Multimodal Informative ViT: Information Aggregation and Distribution for Hyperspectral and LiDAR Classification},
  author={Zhang, Jiaqing and Lei, Jie and Xie, Weiying and Yang, Geng and Li, Daixun and Li, Yunsong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}

@inproceedings{zhange2025DiffCLIP,
  title={DiffCLIP: Few-shot Language-driven Multimodal Classifier },
  author={Zhang, Jiaqing and Cao, Mingxiang and Jiang, Kai and Yang, Xue},
  booktitle={AAAI2025}
}

```
<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="
      https://api.star-history.com/svg?repos=icey-zhang/DiffCLIP&type=Date&theme=dark
    "
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="
      https://api.star-history.com/svg?repos=icey-zhang/DiffCLIP&type=Date
    "
  />
  <img
    alt="Star History Chart"
    src="https://api.star-history.com/svg?repos=icey-zhang/DiffCLIP&type=Date"
    width="600"  <!-- 设置宽度 -->
    height="500" <!-- 可选设置高度 -->
  />
</picture>




