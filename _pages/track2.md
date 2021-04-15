---
layout: page
title: Track 2
---
<p align="center">
    <a href="https://www.kaggle.com/c/dl-hack-track-2-nlp/">COMPETITION LINK</a>
</p>

# Description

Recent improvements in Language models have led to significant progress on various NLP tasks like text classification, question-answering, information-retrieval, etc. Text Generation (or) the task of generating text to appear indistinguishable from human-written text is one such task that has captured the attention of researchers([GPT-3](https://github.com/elyase/awesome-gpt3)).

In this challenge, you will be required to generate abstracts for research papers from their titles. Unlike text summarization, this is a more demanding task as the model needs to predict an entire paragraph from just 5-6 words present in the title, capture cues regarding the progress of research in a specific topic, and most importantly ensure novelty. 

<p align="center">
    <img src="./assets/images/title-abstract.png" width="50%">
</p>

# Dataset

For nearly 30 years, [Arxiv](https://arxiv.org) has served the public and research communities by providing open access to scholarly articles, from the vast branches of physics to the many subdisciplines of computer science to everything in between, including math, statistics, electrical engineering, quantitative biology, and economics. From this collection, papers are selected from the categories: [cs.CV, cs.AI, cs.LG, cs.CL, cs.NE, stat.ML] and their corresponding titles, abstracts are extracted as metadata to build a large corpus containing 1,47,381 entries. The test set contains 1,000 samples for evaluation.

The training and test datasets can be downloaded from [kaggle](https://www.kaggle.com/c/dl-hack-track-2-nlp/data) or from [gdrive](https://drive.google.com/file/d/1TEvk4HHFwMyJyZ0TaKCG6CshlqlbCfi5/view?usp=sharing). For people using collab the dataset can be downloaded on to your local runtime or onto your google drive using the kaggle api:

```python
!pip3 install kaggle

# authenticate using public api token which can found from your kaggle profile page
%env KAGGLE_USERNAME=<username>
%env KAGGLE_KEY=<kaggle-key>

# to download on drive 
# 1) mount drive using the option of the left tab
# 2) %cd drive/MyDrive/ 
!kaggle competitions download -c dl-hack-track-2-nlp
```

# Evaluation

We compute the text embeddings for the predicted abstract (Ref: [SPECTRE](https://arxiv.org/pdf/2004.07180.pdf)) to compare coherence between machine and human generated outputs. Ideally, we would want to compute the cosine similarity (or any other similarity measure) to evaluate the quality of these embeddings. But given the limited number of metrics available on Kaggle, submissions are scored on Root Mean Squared Logarithmic Error (RMSLE) for each index of the text embeddings:

$$ \textrm{RMSLE} = \sqrt{ \frac{1}{N} \sum_{i=1}^N (\log(\hat{y}_i) - \log(y_i))^2 } $$

where
- $$n$$ is the size of the embedding vector
- $$\hat{y}_i$$ is the predicted embedding value at index i
- $$y_i$$ is the actual embedding value at index i
- $$log()$$ is the natural base logarithm
  
A smaller RMSLE is better. It fundamentally computes the relative error and does not vary much with scale, making it as close as possible to mimicing a similarity measure.  

# Submission

SPECTRE produces embeddings of size 768 and to ensure ease in computation, this is reduced to a 32 sized vector using PCA. Hence, the final submission file must contain 33 columns - **"id"**: paper id and columns ranging from **f_0-f_31**: each index of the embedding vector. Sample submission file can be found [here](assets/sample_submission/track2.csv). 

We also provide a submission script which can be found [here](assets/scripts/submit_track2.py) to create embeddings from the predicted abstracts by querying the SPECTRE API. *Here the predicted csvfile corresponds to the model's text predictions which needs to be converted to embeddings.*

**USAGE**
```bash
python3 submit_track2.py <path-to-predicted-csvfile>
```

# IMPORTANT

- Since these papers are available online, there is a possibility of manually extracting the abstracts. Hence, all participants are required to submit their code and any manipulation found in the same will lead to an invalid submission.
- Please make sure to use the provided submission script to create the embeddings for fair comparison with the ground-truth. 
