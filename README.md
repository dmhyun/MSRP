


<h1 align="center">
<span> Generating Multiple-Length Summaries via Reinforcement Learning for Unsupervised Sentence Summarization
</span>
</h1>

<p align="center">
    <a href="https://2022.emnlp.org" alt="Conference">
        <img src="https://img.shields.io/badge/EMNLP'22-Findings-brightgreen" /></a>   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>   
</p>

<p align="center">
<span>Official implementation of </span>
<a href="https://aclanthology.org/2022.findings-emnlp.214/">EMNLP'22 paper</a>
</p>

## Overview

### Unsupervised Summarization

Summarization shortens given texts while maintaining core contents of the texts, and unsupervised approaches have been studied to summarize texts without ground-truth summaries. 

### Reinforcement Learning-based Approach

We devise an **abstractive** model by formulating the summarization task as a reinforcement learning without ground-truth summaries.

<p align="center"><img src="images/model.png" alt="graph" width="99%"></p>

### Summary Accuracy

The proposed model (**MSRP**) substantially outperforms both abstractive and extractive models, yet frequently generating new words not contained in input texts.

<p align="center"><img src="images/accuracy.png" alt="graph" width="65%"></p>

## Major Requirements

* Python

* Pytorch

* transformers

* Numpy

## Evaluation with Trained Models

We uploaded the trained models in HuggingFace library, and you can easily evaluate the uploaded models.

  * anonsubms/msrp_length

  * anonsubms/msrp_ratio

  * anonsubms/msrp_length_sb

  * anonsubms/msrp_ratio_sb


  * <pre><code>python evaulate.py MODEL_ID</code></pre>

  * <pre>[Example] <code>python evaulate.py anonsubms/msrp_length</code></pre>
