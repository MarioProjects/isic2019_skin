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
- [x] Data normalization (255)
- [x] Good Data Sampler
- [ ] More data? [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- [x] [Balanced accuracy score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
- [ ] [Focal Loss](https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/)

### Testing Phases

- [x] PHASE 0 - [LR Finder](https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0) 

- [x] PHASE 1 - Optimization Algorithms
  + [x] Adam
  + [x] SGD Momentum
  + [x] SGD Default
  + [x] RMSprop Momentum
  + [x] RMSprop Default

- [ ] PHASE 2 - Data Augmentation
  + [x] Custom Data Augmentation (based on previous papers)
  + [ ] [Fast Autoaugment (transfer?)](https://arxiv.org/abs/1905.00397)

- [ ] PHASE 3 - Improvement Techniques
  + [x] Weighted Cross Entropy
  + [x] [Color Constancy - Shades of gray and Retinex](http://vislab.isr.ist.utl.pt/wp-content/uploads/2012/12/14-ICIPa.pdf)
  + [x] [Cutout](https://arxiv.org/abs/1708.04552)
  + [x] [Mixup](https://arxiv.org/abs/1710.09412)
  + [x] [CutMix](https://arxiv.org/abs/1905.04899)
  + [x] [Color Spaces - ColorNet](https://arxiv.org/abs/1902.00267)
  + [ ] [Pair Sampling](https://arxiv.org/pdf/1801.02929.pdf)

- [ ] PHASE 4 - [Model Explotation](https://github.com/osmr/imgclsmob/tree/master/pytorch) 
  + [x] Seresnext50
  + [ ] ResnetD-101b

- [ ] PHASE 5 - Last Steps
  + [ ] Try Improve CutMix (See [issues](https://github.com/clovaai/CutMix-PyTorch/issues))
  + [ ] [Snapshot Ensembling](https://arxiv.org/abs/1704.00109)
  + [ ] Test Time Augmentations  
  + [ ] Calibration Methods / Uknown class
  + [ ] Error analysis
  + [ ] Model Ensembling -> Sum / Votation

### Results

#### PHASE 1 . Optimizer testing

|     Optimizer     |            LR Planning               |   Additional Info    |     Accuracy    |  Balanced Accuracy  |
|:-----------------:|:------------------------------------:|:--------------------:|:---------------:|:-------------------:|
|   Adam Decay      |   Constant LR (expertise) 0.001      |  ------------------  |      0.7399     |                     |
|   Adam Decay      |   Constant LR (expertise) 0.001      |      Normalized      |      0.7642     |       0.6500        |
|  ~~Adam Decay~~   |     ~~Step LR (Finder-1exp) 0.01~~   |  ------------------  |   ~~Discarded~~ |                     |
|  ~~SGD Momentum~~ |      ~~Constant LR (Finder) 1~~      |  ------------------  |   ~~Discarded~~ |                     |
|  ~~SGD Momentum~~ |   ~~Constant LR (Finder-1exp) 0.1~~  |  ------------------  |   ~~Discarded~~ |                     |
|  SGD Momentum     |   Constant LR (Expertise) 0.01       |  ------------------  |      0.7470     |                     |
|  SGD Momentum     |      Step LR (Expertise) 0.01        |  ------------------  |      0.7994     |       0.6897        |
|  **SGD Momentum** |    **Step LR (Expertise) 0.01**      |    **Normalized**    |    **0.7992**   |     **0.7000**      |
|  SGD Momentum     |      Step LR (Expertise) 0.01        |   BB / Normalized    |      0.7813     |       0.6700        |
|  SGD Default      |      Constant LR (Finder) 1          |  ------------------  |      0.7859     |                     |
|  SGD Default      |        Step LR (Finder) 1            |  ------------------  |      0.7862     |                     |
|  SGD Default      |        Step LR (Finder) 1            |         BB           |      0.7750     |       0.5862        |
|  **SGD Default**  |      **Step LR (Finder) 1**          | **BB / Normalized**  |    **0.8084**   |     **0.7000**      |

- BB: Batch Balanced
- Normalized: Normalize input images / 255

Conclusions: the optimizer that has worked best has generally been SGD, 
so we will choose this for the following phases. 
In addition, for the models that have been trained in this phase **we have 
forgotten to normalize** (Normalize) the images... corrected for next phases!
**Balanced sampler has no clear impact** on results...

#### PHASE 2 . Data Augmentation

|     Optimizer     |            LR Planning               |      Additional Info      |     Accuracy    |  Balanced Accuracy  |
|:-----------------:|:------------------------------------:|:-------------------------:|:---------------:|:-------------------:|
|   SGD Momentum    |      Step LR (Expertise) 0.01        |     ------------------    |      0.8226     |       0.7320        |
|   SGD Momentum    |      Step LR (Expertise) 0.01        |          Aggro DA         |      0.8244     |       0.7306        |

In this experiments all images are normalized dividing the data by 255. Custom Data Augmentation employed:

```python
# Custom Data Augmentation
train_aug = albumentations.Compose([
    albumentations.PadIfNeeded(p=1, min_height=args.crop_size, min_width=args.crop_size),
    albumentations.Resize(args.img_size, args.img_size),
    albumentations.RandomCrop(p=1, height=args.crop_size, width=args.crop_size),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.RandomBrightnessContrast(p=0.5, brightness_limit=0.22, contrast_limit=0.22),
    albumentations.HueSaturationValue(p=0.5, hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5),
    albumentations.ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=0.1, rotate_limit=45)
])

# Custom Aggro Data Augmentation
train_aug = albumentations.Compose([
    albumentations.PadIfNeeded(p=1, min_height=args.crop_size, min_width=args.crop_size),
    albumentations.Resize(args.img_size, args.img_size),
    albumentations.RandomCrop(p=1, height=args.crop_size, width=args.crop_size),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.RandomBrightnessContrast(p=0.5, brightness_limit=0.25, contrast_limit=0.25),
    albumentations.HueSaturationValue(p=0.5, hue_shift_limit=9, sat_shift_limit=14, val_shift_limit=10),
    albumentations.ShiftScaleRotate(p=0.5, shift_limit=0.2, scale_limit=0.2, rotate_limit=75)
])
```

 
#### PHASE 3 - Improvement Techniques 

For Mixup we have tested $\alpha$=0.5 and $\alpha$=0.2 meanwhile for cutout mask size 50x50.

|     Optimizer     |              LR Planning             |          Additional Info           |     Accuracy    |  Balanced Accuracy  |
|:-----------------:|:------------------------------------:|:----------------------------------:|:---------------:|:-------------------:|
|   SGD Momentum    |        Step LR (Expertise) 0.01      |              Cutout50              |      0.8247     |       0.7517        |
|   SGD Momentum    |        Step LR (Expertise) 0.01      |  Cutout50 - Weighted Loss Subtract |      0.8128     |       0.7478        |
|   SGD Momentum    |        Step LR (Expertise) 0.01      |              Retinex               |      0.8055     |       0.7057        |
|   SGD Momentum    |        Step LR (Expertise) 0.01      |  ShadesGray - NoColorTrans         |      0.8073     |       0.7253        |
|   SGD Momentum    |        Step LR (Expertise) 0.01      |              ColorNet              |      0.8000     |       0.6813        |

#### PHASE 4 - Model Explotation

|     Optimizer     |            LR Planning               |      Model     |            Additional Info          |     Accuracy    |  Balanced Accuracy  |
|:-----------------:|:------------------------------------:|:--------------:|:-----------------------------------:|:---------------:|:-------------------:|
|   SGD Momentum    |      Step LR (Expertise) 0.01        |   SeresNext50  |   Cutout50 - Weighted Loss Subtract |      0.8763     |       0.8109        |
|   SGD Momentum    |      Step LR (Expertise) 0.01        |   Resnetd101b  |   Cutout50 - Weighted Loss Subtract |      0.8813     |       0.8298        |


### Reminders
- Confusion Matrix
- Pretrained Model Test
- Old Comeptitions Results
