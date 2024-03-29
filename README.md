# Ensemble model adversarial attack

This project explores the transferability of adversarial examples in RobustBench models.
The code can also be readily used in the following colab link to quickly reproduce the same results provided:
https://colab.research.google.com/drive/1ajLYUpvWYhVZacY46wCZ4T-noT-CP1zn?usp=sharing

## Overview

The goal of this project is to study how adversarial examples crafted to deceive an ensemble of models can also fool other more robust models.
The adversarial examples are crafted using Fast Gradient Sign Method (FGSM) on a set of 3 RobustBench models trained with CIFAR-10 data and L-inf norm perturbations (ε = 8/255).

## Key Features

- Transferability analysis of adversarial examples across multiple models
- Crafting ensemble model adversarial examples using FGSM
- Evaluation of success rates and average test error rates on CIFAR-10 dataset
- Identification of clustering tendencies in adversarial perturbations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/TrueSalmonKing/Ensemble_model_adversarial_attack.git
```
   
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script to conduct the transferability analysis in addition to viewing the visualization of the perturbations added
```bash
python main.py
```
