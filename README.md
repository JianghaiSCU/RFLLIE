# [CVIU 2024]Low-light Image Enhancement with Wavelet-based Diffusion Models [[Paper]]().
<h4 align="center">Hai Jiang<sup>1</sup>, Yang Ren<sup>1</sup>, Songchen Han<sup>1,†</sup></sup></center>
<h4 align="center"><sup>1.</sup>Sichuan University</center>

## Pipeline
![](./Figures/pipeline.png)

## Dependencies
```
pip install -r requirements.txt
````

## Download the raw training and evaluation datasets
### Paired datasets 
LOLv1 dataset: Chen Wei, Wenjing Wang, Wenhan Yang, and Jiaying Liu. "Deep Retinex Decomposition for Low-Light Enhancement", BMVC, 2018. [[Baiduyun (extracted code: sdd0)]](https://pan.baidu.com/s/1spt0kYU3OqsQSND-be4UaA) [[Google Drive]](https://drive.google.com/file/d/18bs_mAREhLipaM2qvhxs7u7ff2VSHet2/view?usp=sharing)

LOLv2 dataset: Wenhan Yang, Haofeng Huang, Wenjing Wang, Shiqi Wang, and Jiaying Liu. "Sparse Gradient Regularized Deep Retinex Network for Robust Low-Light Image Enhancement", TIP, 2021. [[Baiduyun (extracted code: l9xm)]](https://pan.baidu.com/s/1U9ePTfeLlnEbr5dtI1tm5g) [[Google Drive]](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view?usp=sharing)

LSRW dataset: Jiang Hai, Zhu Xuan, Ren Yang, Yutong Hao, Fengzhu Zou, Fang Lin, and Songchen Han. "R2RNet: Low-light Image Enhancement via Real-low to Real-normal Network", Journal of Visual Communication and Image Representation, 2023. [[Baiduyun (extracted code: wmrr)]](https://pan.baidu.com/s/1XHWQAS0ZNrnCyZ-bq7MKvA)

### Unpaired datasets 
Please refer to [[Project Page of RetinexNet.]](https://daooshee.github.io/BMVC2018website/)

## Pre-trained Models 
You can downlaod our pre-trained model from [[Google Drive]]() and [[Baidu Yun (extracted code:)]]()

## How to train?
```
python main.py --mode train
```

## How to test?
```
python main.py --mode test
```

## Visual comparison
![](./Figures/comparison.png)

## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```

```

## Acknowledgement
Part of the code is adapted from the previous work: [MIMO-UNet](https://github.com/chosj95/MIMO-UNet). We thank all the authors for their contributions.
