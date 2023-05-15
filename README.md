# S2PLN

The code repo for the paper "Learning a Spoof Pattern Generation Network with Identity Disentanglement for Semi-Supervised Cross-Domain Face Anti-Spoofing"


## Congifuration Environment
- python 3.7 
- pytorch 1.7.1
- torchvision 0.8.2
- cuda 11.0

## Pre-processing
### **Dataset** 
Download the **OULU-NPU(O)**, **CASIA-FASD(C)**, **Idiap Replay-Attack(I)**, **MSU-MFSD(M)**, **MS-Celeb-
1M** and **LFW** datasets


### **Face Detection and Face Alignment** 

All video frames of the four datasets are pre-processed with the [MTCNN algorithm](https://ieeexplore.ieee.org/abstract/document/7553523) to perform face detection and cropping, and the cropped RGB face images are resized to 224 × 224 × 3.

### **Label Generation** 

```python
python generate_label.py
```

### **Training** 




                                                                            