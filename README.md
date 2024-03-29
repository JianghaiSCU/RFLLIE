# [CVIU 2024] Revisiting Coarse-to-fine Strategy for Low-Light Image Enhancement with Deep Decomposition Guided Training [[Paper]](https://www.sciencedirect.com/science/article/pii/S107731422400033X).
<h4 align="center">Hai Jiang<sup>1</sup>, Yang Ren<sup>1</sup>, Songchen Han<sup>1,†</sup></sup></center>
<h4 align="center"><sup>1</sup>School of Aeronautics and Astronautics, Sichuan University</center>

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

## Pipeline
![](./Figures/pipeline.jpg)

## Pre-trained Models 
You can downlaod our pre-trained model from [[Google Drive]](https://drive.google.com/drive/folders/1sKjGlWJt2sxHkDApymv0Tgbg4R1Q1ZCb?usp=sharing) and [[Baidu Yun (extracted code:mrzz)]](https://pan.baidu.com/s/1JOL2xlKqoaH4e8Z_fAkalQ )

## How to train?
```
python main.py --mode train
```

## How to test?
```
python main.py --mode test
```

## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```
@article{jiang2023low,
  title={Revisiting Coarse-to-fine Strategy for Low-Light Image Enhancement with Deep Decomposition Guided Training},
  author={Jiang, Hai and Ren, Yang and Han, Songchen},
  journal={Computer Vision and Image Understanding},
  volume = {241},
  pages = {103952},
  year = {2024}
}
```

## Acknowledgement
Part of the code is adapted from the previous work: [MIMO-UNet](https://github.com/chosj95/MIMO-UNet). We thank all the authors for their contributions.
