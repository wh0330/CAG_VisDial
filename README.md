Iterative Context-Aware Graph Inference for Visual Dialog
====================================


![alt text](https://github.com/wh0330/CAG_VisDial/blob/master/image/framework.png)
<p align="center">The overall framework of Context-Aware Graph.</p>



This is a PyTorch implementation for [Iterative Context-Aware Graph Inference for Visual Dialog, CVPR2020](https://arxiv.org/abs/2004.02194).


If you use this code in your research, please consider citing:

```text
@InProceedings{Guo_2020_CVPR,
author = {Guo, Dan and Wang, Hui and Zhang, Hanwang and Zha, Zheng-Jun and Wang, Meng},
title = {Iterative Context-Aware Graph Inference for Visual Dialog},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}

```

Requirements
----------------------
This code is implemented using PyTorch v0.3.1, and provides out of the box support with CUDA 9 and CuDNN 7. 


Data
----------------------

1. Download the VisDial v1.0 dialog json files and images from [here][1].
2. Download the word counts file for VisDial v1.0 train split from [here][2]. 
3. Use Faster-RCNN to extract image features from [here][3].
4. Download pre-trained GloVe word vectors from [here][4].


Training
--------

Train the CAG model as:

```sh
python train/train_D_1.0.py --CUDA
```


Evaluation
----------

Evaluation of a trained model checkpoint can be done as follows:

```sh
python eval/evaluate.py --model_path [path_to_root]/save/XXXXX.pth --cuda
```
This will generate an EvalAI submission file, and you can submit the json file to [online evaluation server][5] to get the result on v1.0 test-std.

  Model  |  NDCG   |  MRR   |  R@1  | R@5  |  R@10   |  Mean  |
 ------- | ------ | ------ | ------ | ------ | ------ | ------ |
CAG | 56.64 | 63.49 | 49.85 |  80.63| 90.15 | 4.11 |

Acknowledgements
----------------

* This code began with [jiasenlu/visDial.pytorch][6]. We thank the developers for doing most of the heavy-lifting.


[1]: https://visualdialog.org/data
[2]: https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/visdial_1.0_word_counts_train.json
[3]: https://github.com/peteanderson80/bottom-up-attention
[4]: https://github.com/stanfordnlp/GloVe
[5]: https://evalai.cloudcv.org/web/challenges/challenge-page/161/overview
[6]: https://github.com/jiasenlu/visDial.pytorch
