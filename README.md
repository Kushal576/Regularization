# Regularization

This repository demonstrates the implementation of **L1 (Lasso)** and **L2 (Ridge)** regularization from scratch for linear regression, as part of an Artificial Intelligence course project. The project explores how regularization helps reduce overfitting and improve model generalization.

---

## Table of Contents

- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

---

## Overview

Regularization is a core technique in machine learning for preventing overfitting by adding a penalty to the loss function. This project implements both L1 and L2 regularization for linear regression and visualizes their effects on model weights and predictions.

---

## Key Concepts

- **L1 Regularization (Lasso):** Adds the sum of absolute values of weights to the loss function. Encourages sparsity (some weights become zero).
- **L2 Regularization (Ridge):** Adds the sum of squared weights to the loss function. Encourages smaller weights but not necessarily zero.
- **Overfitting:** Model fits training data too closely, performs poorly on new data.
- **Generalization:** Model's ability to perform well on unseen data.

---

## Project Structure

Regularization/
├── data/ # Sample datasets
├── model/
├──── l1_regularization.py # L1 implementation
├──── l2_regularization.py # L2 implementation
├──── other implementation files
├── README.md # Project documentation


## Usage

1. **Clone the repository:**
git clone https://github.com/Kushal576/Regularization.git
cd Regularization


3. **Run the scripts:**
python l1_regularization.py
python l2_regularization.py


4. **View results:**  
Outputs graphs are directly obtained.


## Results

- Regularization reduces model coefficient magnitudes and improves test performance.
- Visualizations show the effect of different regularization strengths (`lambda`) on weights and predictions.
- L1 regularization can produce sparse solutions (some coefficients exactly zero).


## References

- [Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville](https://www.deeplearningbook.org/)
- [Scikit-learn documentation: Regularization](https://scikit-learn.org/stable/modules/linear_model.html#regularization)
- [Regularization Techniques (GitHub)](https://github.com/ashishpatel26/Regularization-Collection-Deeplearning)



