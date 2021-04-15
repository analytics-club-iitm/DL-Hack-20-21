---
layout: page
title: Track 1
---
<p align="center">
    <a href="https://www.kaggle.com/c/dl-hack-track-1-cv/">COMPETITION LINK</a>
</p>

# Description

Recent progress in Computer Vision (specifically in Generative models) has enabled the creation of high-resolution deep fakes which are practically indistinguishable from real ones. However, when this technology falls into the wrong hands, it can be used maliciously as a source of misinformation, manipulation, harassment, and persuasion. Identifying manipulated media is a technically demanding and rapidly evolving challenge.

In this challenge, you will be required to classify images as "real" or "fake". The goal of this challenge is to spur students to build new and innovative solutions that can help detect deep fakes and manipulated media. 
<p align="center">
    <img src="./assets/images/deepfake-cls.png" width="40%">
</p>

# Dataset

The train set contains 35,000 high resolution (512x512) images equally split into and real and fake samples. The test set contains 5,000 images of the same resolution.

The training and test datasets can be downloaded from [kaggle](https://www.kaggle.com/c/dl-hack-track-1-cv/data) or from [gdrive](https://drive.google.com/file/d/1vIHhU2rPw5yqBjePYcMrJs494mUfUTd7/view?usp=sharing). For people using collab the dataset can be downloaded on to your local runtime or onto your google drive using the kaggle api:

```python
!pip3 install kaggle

# authenticate using public api token which can found from your kaggle profile page
%env KAGGLE_USERNAME=<username>
%env KAGGLE_KEY=<kaggle-key>

# to download on drive 
# 1) mount drive using the option of the left tab
# 2) %cd drive/MyDrive/ 
!kaggle competitions download -c dl-hack-track-1-cv
```
## Samples from the dataset

<div style="float:left;margin-right:5px;width:45%">
    <img src="./assets/images/fake-sample.png" width="100%"/>
    <p style="text-align:center;">Fake Samples</p>
</div>
<div style="float:right;margin-right:5px;width:45%">
    <img src="./assets/images/real-sample.png"  width="100%"/>
    <p style="text-align:center;">Real Samples</p>
</div>

# Evaluation

Submissions are scored on log loss:

$$\textrm{LogLoss} = - \frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\right],$$

where
- $$n$$ is the number of images being predicted
- $$\hat{y}_i$$ is the predicted probability of the image being **REAL**
- $$y_i$$ is 1 if the image is **REAL**, 0 if **FAKE**
- $$log()$$ is the natural base logarithm
  
A smaller log loss is better. The use of the logarithm provides extreme punishments for being both confident and wrong. 

# Submission

The final submission must contain two columns - **"id"**: image id and **"p_real"**: predicted probability of the image being real. Sample submission file can be found [here](assets/sample_submission/track1.csv). 

We also provide a submission script which can be found [here](assets/scripts/submit_track1.py). *Please make the required changes to incorporate your model into the submission scripts (there are detailed comments provided for your reference). NOTE: Normalization is performed after the output is derived from the model (lines: 81-87).* 

**USAGE**
```bash
python3 submit_track1.py <path-to-test-datadir> <model-ckpt>
```

# IMPORTANT

- Please note that the submission should be the **PROBABILITY** that the given image is real and not the class labels. (refer to submission script [here](assets/scripts/submit_track1.py))
- The generator used to create the test set is specifically fine-tuned to spoof deep fake detectors. Hence a high-performing model on the train set **NEED NOT** give a similar performance in the test set.
- During inference, you are **NOT** allowed to resize the images in the test set. Any violation of this rule will lead to an invalid submission.
- All participants are required to submit their code and any manipulation found in the same will lead to an invalid submission.
- **SUGGESTION**: As mentioned before log loss penalizes the model for being both confident and wrong. In the worst possible case, a prediction that something is TRUE when it is FALSE will add an infinite amount to your error score. Hence, it's better to bound your predictions from the extremes by a small value. This can be done simply by:
```python
import numpy as np
# here prob_real is the predicted probabilty of the image being REAL
# where upper and lower bounds can be 0.025 and 0.975
prob_real = np.clip(prob_real, <lower-bound>, <upper-bound>)
```
