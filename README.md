GAN Stability: Extragradient vs Optimistic Gradient

Repository for experiments from the SPbU research practice report:

“Extragradient and Optimistic Methods in GAN Training:
Variational Inequalities, Monotonicity and Stability of Adversarial Learning”

Overview

This repository contains implementations and experiments comparing:

* Vanilla Gradient Descent-Ascent (GDA)
* Extragradient Method (Korpelevich correction)
* Optimistic Gradient Descent (OGDA)

for stabilizing GAN training on toy 2D Gaussian mixture datasets.



Installation

Clone repository:

git clone https://github.com/USERNAME/gan-stability-convergence-plots.git
cd gan-stability-convergence-plots

Install dependencies:

pip install -r requirements.txt

Run experiments

Vanilla GDA:

python src/train_vanilla.py

Extragradient:

python src/train_extragradient.py

Optimistic GD:

python src/train_optimistic.py

Run full experiment suite:

python experiments/run_all.py

Reproducibility

Fixed random seed is used.

Toy dataset:
8 Gaussian mixture on a circle (2D).

Results

Example trajectory and Jacobian spectrum:

Citation

SPbU Research Practice Report, 2026.
