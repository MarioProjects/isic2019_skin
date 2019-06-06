# ISIC 2019: Skin Lesion Analysis Towards Melanoma Detection

## Task
The goal for [ISIC 2019](https://challenge2019.isic-archive.com/) is classify dermoscopic images 
among nine different diagnostic categories:

+ Melanoma
+ Melanocytic nevus
+ Basal cell carcinoma
+ Actinic keratosis
+ Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)
+ Dermatofibroma
+ Vascular lesion
+ Squamous cell carcinoma
+ None of the others

25,332 images are available for training across 8 different categories. Additionally, the test dataset 
(planned release August 2nd) will contain an additional outlier class not represented in the training data, 
which developed systems must be able to identify.

Two tasks will be available for participation: 1) classify dermoscopic images without meta-data, and 
2) classify images with additional available meta-data. Task 1's deadline will be August 9th. 
Task 2 will be August 16th, after release of test meta-data on August 9th. 
Participants of Task 2 must submit to Task 1 as well, though participants can submit to Task 1 alone.

In addition to submitting predictions, each competitor is required to submit a link to a manuscript 
describing the methods used to generate predictions.


## Timeline
1. Training data release May 3, 2019
2. Test images release August 2, 2019
3. Submission deadline: images-only test August 9, 2019
4. Test metadata release August 9, 2019
5. Submission deadline: images and metadata test August 16, 2019
6. Winners announced, and speaker invitations sent August 23, 2019
7. MICCAI2019 Workshop October 13 or 17, 2019

# This Repo


This repository aims to address the challenge proposed in the International Skin Imaging Collaboration (ISIC):
Skin Lesion Analysis Towards Melanoma Detection.

 
### Reproducibility

If you want to be able to work with the techniques and models discussed here, 
it is necessary to install certain packages. To facilitate this, the use of Docker is proposed:

```bash
ToDo
```

### ToTest Techniques

Some techniques that we are using or should be used, as a reminder, are:
- [ ] Good Data Sampler
- [ ] AUC Cost Function (If AUC as main Metric)
- [ ] Special Data Augmentation 

### Testing Phases

- [x] [LR Finder](https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0) 

- [ ] Optimization Algorithms
  + [ ] Adam
  + [ ] SGD Momentum
  + [ ] SGD Default
  + [ ] RMSprop Momentum
  + [ ] RMSprop Default

- [ ] Data Augmentation
  + [ ] [Fast Autoaugment (transfer?)](https://arxiv.org/abs/1905.00397)

- [ ] [EfficientNet Explotation](https://arxiv.org/pdf/1905.11946.pdf) 
  + [ ] Coefficients Search
  + [ ] Scale EfficientNet 
  + [ ] Compare with powerful Model (ResNet?)

- [ ] Improvement Techniques
  + [ ] [Pair Sampling](https://arxiv.org/pdf/1801.02929.pdf)
  + [ ] [Snapshot Ensembling](https://arxiv.org/abs/1704.00109)
  + [ ] Test Time Augmentations
  + [ ] Calibration Methods  
  

### Results

#### PHASE 1 . Optimizer testing

|     Optimizer     |            LR Planning               |       Results       |
|:-----------------:|:------------------------------------:|:-------------------:|
|   Adam Decay      |   Constant LR (expertise) 0.001      |       0.739932      |
|   Adam Decay      |   Constant LR (Finder-1exp) 0.01     |        ToRun        |
|  Adam Default     |        Step LR (Finder) 0.1          |        ToRun        |
| Adam Nesterov     |        Step LR (Finder) 0.1          |        ToRun        |
|  ~~SGD Momentum~~ |      ~~Constant LR (Finder) 1~~      |    ~~Discarded~~    |
|  ~~SGD Momentum~~ |   ~~Constant LR (Finder-1exp) 0.1~~  |    ~~Discarded~~    |
|  SGD Momentum     |   Constant LR (Expertise) 0.01       |       0.747039      |
|  SGD Momentum     |      Step LR (Expertise) 0.01        |    Running gpu21    |
|  SGD Default      |      Constant LR (Finder) 1          |    Running gpu11    |
|  SGD Default      |        Step LR (Finder) 1            |    Running gpu18    |

### Reminders
- Confusion Matrix
- Pretrained Model Test
- Old Comeptitions Results