# Pytorch version of End2end image reconstruction

---

This is a Pytorch implementation of [Shen, Dwivedi, Majima, Horikawa, and Kamitani (2019) End-to-end deep image reconstruction from human brain activity. Front. Comput. Neurosci](https://www.frontiersin.org/articles/10.3389/fncom.2019.00021/full)

For Caffe version, see:
https://github.com/KamitaniLab/End2EndDeepImageReconstruction


The comparator is slightly different from the one in Caffe version, here AlexNet model is used:
https://pytorch.org/vision/stable/_modules/torchvision/models/alexnet.html


## Requirements

---

- pytorch
- torchvision
- Python 3
- Numpy
- Scipy
- Pillow (PIL)

## Data

---

Please refer to https://github.com/KamitaniLab/End2EndDeepImageReconstruction/tree/master/data

## Usage

---

For training a new model, run
`python end2end_train.py`

For testing, run
`python end2end_test.py`


## Notes

---

Because the data size is not large, the current code loads all fMRI data and all images at once.

TODO: Load data on the fly.

## References

---

[1] We used the framework proposed in this article: Dosovitskiy & Brox (2016) Generating Images with Perceptual Similarity Metrics based on Deep Networks. Advances in Neural Information Processing Systems (NIPS).

The article is available at: http://arxiv.org/abs/1602.02644

[2] Shen, Horikawa, Majima, and Kamitani (2019) Deep image reconstruction from human brain activity, http://dx.doi.org/10.1371/journal.pcbi.1006633