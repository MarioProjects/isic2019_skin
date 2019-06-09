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
it is necessary to install certain packages. To facilitate this, the use of Docker is proposed. 
First, access the docker folder of this repository and execute the following command:

```bash
sudo ./reproduce.sh
```

This will download the ISIC data into your $ {HOME} / data folder and clone this repository to $ {HOME}. 
Finally, you will create an image with pytorch and the necessary packages to use the models proposed here 
and start that image, assembling the data and the repository in the home.

### ToTest Techniques

Some techniques that we are using or should be used, as a reminder, are:
- [ ] Good Data Sampler
- [ ] AUC Cost Function (If AUC as main Metric)
- [ ] Special Data Augmentation 

### Testing Phases

- [x] PHASE 0 - [LR Finder](https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0) 

- [x] PHASE 1 - Optimization Algorithms
  + [x] Adam
  + [x] SGD Momentum
  + [x] SGD Default
  + [x] RMSprop Momentum
  + [x] RMSprop Default

- [ ] PHASE 2 - Data Augmentation
  + [ ] [Fast Autoaugment (transfer?)](https://arxiv.org/abs/1905.00397)

- [ ] PHASE 3 - [EfficientNet Explotation](https://arxiv.org/pdf/1905.11946.pdf) 
  + [ ] Coefficients Search
  + [ ] Scale EfficientNet 
  + [ ] Compare with powerful Model (ResNet?)

- [ ] PHASE 4 - Improvement Techniques
  + [ ] [Pair Sampling](https://arxiv.org/pdf/1801.02929.pdf)
  + [ ] [Snapshot Ensembling](https://arxiv.org/abs/1704.00109)
  + [ ] Test Time Augmentations
  + [ ] Calibration Methods  
  

### Results

#### PHASE 1 . Optimizer testing

|     Optimizer     |            LR Planning               |   Additional Info    |       Results   |
|:-----------------:|:------------------------------------:|:--------------------:|:---------------:|
|   Adam Decay      |   Constant LR (expertise) 0.001      |  ------------------  |      0.739932   |
|  ~~Adam Decay~~   |     ~~Step LR (Finder-1exp) 0.01~~   |  ------------------  |   ~~Discarded~~ |
|  ~~SGD Momentum~~ |      ~~Constant LR (Finder) 1~~      |  ------------------  |   ~~Discarded~~ |
|  ~~SGD Momentum~~ |   ~~Constant LR (Finder-1exp) 0.1~~  |  ------------------  |   ~~Discarded~~ |
|  SGD Momentum     |   Constant LR (Expertise) 0.01       |  ------------------  |      0.747039   |
|  SGD Momentum     |      Step LR (Expertise) 0.01        |  ------------------  |      0.799473   |
|  SGD Default      |      Constant LR (Finder) 1          |  ------------------  |      0.785996   |
|  SGD Default      |        Step LR (Finder) 1            |  ------------------  |      0.786260   |
|  SGD Default      |        Step LR (Finder) 1            | Balanced Dataloader  |   Running gpu11 | 


Sonclusions: the optimizer that has worked best has generally been SGD, 
so we will choose this for the following phases.

### Reminders
- Confusion Matrix
- Pretrained Model Test
- Old Comeptitions Results
