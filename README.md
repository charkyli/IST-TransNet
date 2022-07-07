# IST-TransNet
# IST-TransNet
![](https://img.shields.io/badge/language-PyTorch-blue.svg?style=flat-square)
[![](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](./LICENSE)

[comment]: <> (> [Paper Link]&#40;https://arxiv.org/abs/2111.03580&#41;   )

[comment]: <> (> Authors: Tianfang Zhang, Siying Cao, Tian Pu and Zhenming Peng  )
##Our network architecture
![network](figure/network.png)
## [Datasets](#IST-TransNet)
- The SIRST Augment dataset: download from [Google Drive](https://drive.google.com/file/d/13hhEwYHU19oxanXYf-wUpZ7JtiwY8LuT/view?usp=sharing) or [BaiduYun Drive](https://pan.baidu.com/s/1c35pADjPhkAcLwmU-u0RBA) with code `ojd4`.

## [Usage](#IST-TransNet)

### Train
```
python train.py --net-name transnet_1 --batch-size 8 --save-iter-step 40 --dataset sirstaug
```

### Inference

```
python inference.py --pkl-path {checkpoint path} --image-path {image path}
```

### Evaluation
```
python evaluation.py --dataset {dataset name} 
                     --sirstaug-dir {base dir of sirstaug}
                     --pkl-path {checkpoint path}
```


## [Results](#IST-TransNet)





Evaluation of model-driven algorithms based on traditional metrics refers [ISTD-python].
