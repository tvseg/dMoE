# Official source code for Distribution-aware Fairness Learning in Medical Image Segmentation From A Control-Theoretic Perspective

Authors: Yujin Oh*, Pengfei Jin*, Sangjoon Park*, Sekeun Kim, Siyeop Yoon, Kyungsang Kim, Jin Sung Kim, Xiang Li, Quanzheng Li

*Equal contribution

https://arxiv.org/abs/2502.00619

---

> **Abstract:** *TEnsuring fairness in medical image segmentation is critical due to biases in imbalanced clinical data acquisition caused by demographic attributes (e.g., age, sex, race) and clinical factors (e.g., disease severity). To address these challenges, we introduce Distribution-aware Mixture of Experts (dMoE), inspired by optimal control theory. We provide a comprehensive analysis of its underlying mechanisms and clarify dMoE's role in adapting to heterogeneous distributions in medical image segmentation. Furthermore, we integrate dMoE into multiple network architectures, demonstrating its broad applicability across diverse medical image analysis tasks. By incorporating demographic and clinical factors, dMoE achieves state-of-the-art performance on two 2D benchmark datasets and a 3D in-house dataset. Our results highlight the effectiveness of dMoE in mitigating biases from imbalanced distributions, offering a promising approach to bridging control theory and medical image segmentation within fairness learning paradigms. 
>
> <p align="center">
> <img width="800" src="main.png">
> </p>

## News
* [2025.05.01] Our paper is accepted on ICML2025, Spotlight Poster. 

## Pretrained models:

|     Dataset     |   Setting    |    Model Zoo   |
| :-------------: | :---------:  | :----------------------------------------------------------: |
| 2D FairSeg |  dMoE   |    |
| 2D HAM10000 | dMoE |   |
| 3D Radiotherapy Target | dMoE |   |

## Inference:
 `bash test.sh`
 `bash test_ham.sh`

## Acknowledgement:
 - FairSeg:  [https://github.com/Harvard-Ophthalmology-AI-Lab/FairSeg]
 - HAM10000: [https://datasetninja.com/skin-cancer-ham10000]
 
## Citation:
```
@misc{oh2025distributionawarefairnesslearningmedical,
      title={Distribution-aware Fairness Learning in Medical Image Segmentation From A Control-Theoretic Perspective}, 
      author={Yujin Oh and Pengfei Jin and Sangjoon Park and Sekeun Kim and Siyeop Yoon and Kyungsang Kim and Jin Sung Kim and Xiang Li and Quanzheng Li},
      year={2025},
      eprint={2502.00619},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2502.00619}, 
}
```
