# HOPE: Hybrid Orbit Prediction Engine (with plans for Explainability)

**Project Type:** Hybrid orbit prediction system combining classical astrodynamics and machine learning  
**Author:** Rajrupa Das  
**Status:** Simulation and dataset generation complete; machine learning model development in progress

---

## Overview

HOPE is a hybrid orbit prediction engine that simulates and forecasts the motion of Low Earth Orbit (LEO) satellites under perturbative forces. The initial version focuses on the J2 zonal harmonic and combines traditional physics-based propagation with machine learning models such as LSTMs. The goal is to evaluate the effectiveness of ML in orbital prediction while maintaining physical consistency and enabling future explainability.

---

## Motivation

Classical numerical propagators (e.g., RK4) provide accurate orbital predictions but are often computationally intensive, especially for long-duration or multi-object simulations. In contrast, machine learning offers a faster alternative but lacks inherent physical interpretability. This project explores a middle ground by using classical propagation to generate training data for ML models, aiming to create a hybrid system that is both efficient and physically grounded.

---

## Features and Progress

| Step      | Description                                                             | Status       |
|-----------|-------------------------------------------------------------------------|--------------|
| Step 1    | Orbit simulation using Poliastro with J2 perturbation                  | Completed    |
| Step 2    | Time-series dataset preparation for ML (sliding window, normalization) | Completed    |
| Step 3    | Build and train LSTM model to predict future orbital positions         | In Progress  |
| Step 4    | Compare ML predictions with RK4 propagation (baseline)                 | Pending      |
| Step 5    | Evaluate hybrid model (average RK4 + LSTM) and visualize errors        | Pending      |

---

## Repository Structure

├── data/ # Raw and preprocessed data
│ ├── orbit_j2.csv # Simulated J2 orbit data
│ ├── X_train.npy # LSTM training input
│ ├── y_train.npy # LSTM training labels
│ ├── lstm_predictions.npy # LSTM model predictions
│ └── position_scaler.save # Scaler for normalizing input
├── src/ # Source code
│ ├── simulate_orbit.py # Orbit simulation script
│ ├── prepare_dataset.py # Time-series dataset generation
│ ├── lstm_model.py # LSTM model definition and training
│ └── plot_predictions.py # Visualization and comparison plots
├── results/ # Output plots and evaluation metrics
├── research_summary.md # Technical notes and research documentation
├── requirements.txt # Python dependencies
└── README.md # Project overview


---

## Setup and Usage

Install dependencies:

```bash
pip install -r requirements.txt
python3 src/prepare_dataset.py
python3 src/lstm_model.py
python3 src/plot_predictions.py

