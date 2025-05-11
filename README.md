# Tahoeformer

## Members
- Ryan Keivanfar [GitHub: rylosqualo](https://github.com/rylosqualo)
- Min Dai [GitHub: genecell](https://github.com/genecell)
- Xinyu Yuan [GitHub: KatarinaYuan](https://github.com/KatarinaYuan)
- Qichen Huang [GitHub: qhuang20](https://github.com/qhuang20)


## Project

### Title
Tahoeformer: Interpreting Cellular Context and DNA Sequence Determinants Underlying Drug Response

### Overview
Tahoeformer is a deep learning model that integrates cellular context and DNA sequence information to predict drug responses. Built upon the Enformer architecture, our model aims to understand how genome variations influence drug effects in different cellular environments.

### Motivation
Precision medicine requires understanding how genetic variations affect drug responses across different cellular contexts. Tahoeformer addresses this challenge by modeling:
- Cellular context (different transcriptional factor expression patterns)
- DNA sequence variations (transcriptional factor binding site mutations)

### Methods
We fine-tuned the Enformer architecture using the Tahoe-100M dataset, incorporating:
- Morgan fingerprints for drug representation
- Pseudobulked gene expression data across 8 cell lines with 27 drugs at a single dosage
- DNA sequence information centered around TSS (transcription start sites) from a curated subset of 500 genes

### Results
Our model demonstrates strong performance on top 20 curated genes in predicting gene expression changes in response to drug treatments across different cellular contexts, enabling better understanding of drug-genome interactions. 



## Code
- [Tahoeformer](https://github.com/genecell/Tahoeformer)

## HuggingFace
- [Tahoeformer](https://huggingface.co/qhuang20/Tahoeformer)

## Datasets
- [Tahoe-100M dataset](https://huggingface.co/datasets/tahoebio/Tahoe-100M)

## Acknowledgements
- [Enformer](https://www.nature.com/articles/s41592-021-01252-x)
- [GradientShap](https://captum.ai/api/gradient_shap.html#) 
- [Weights & Biases](https://wandb.ai/)

