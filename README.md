# S2PLN

The code repo for the paper "Learning a Spoof Pattern Generation Network with Identity Disentanglement for Semi-Supervised Cross-Domain Face Anti-Spoofing"

The motivation of the proposed S2PLN method:


## Table of Contents
- [Congifuration Environment](##Congifuration Environment)
- [Pre-processing](##Pre-processing)
- [Training](##Training)

## Congifuration Environment
- python 3.7 
- pytorch 1.7.1
- torchvision 0.8.2
- cuda 11.0

### Installation
To install the required packages, you can run `pip install -r requirements.txt`

## Pre-processing
### **Dataset** 
Download the **OULU-NPU(O)**, **CASIA-FASD(C)**, **Idiap Replay-Attack(I)**, **MSU-MFSD(M)**, **MS-Celeb-
1M** and **LFW** datasets

### **Face Detection and Face Alignment**
All video frames of the four datasets are pre-processed with the [MTCNN algorithm](https://ieeexplore.ieee.org/abstract/document/7553523) to perform face detection and cropping, and the cropped RGB face images are resized to 224 × 224 × 3.

### **Label Generation**
To generate the data label list, you can run the following command:
```python
python datalabel/generate_label.py
```

## **Training** 


## Acknowledgments
We would like to express our gratitude to [SSDG](https://github.com/taylover-pei/SSDG-CVPR2020), which are of great help to our work.

                                                                            