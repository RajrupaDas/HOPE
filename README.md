# HOPE: Hybrid Orbit Prediction Engine (with plans for Explainability) 

**Project Type:** Hybrid orbit prediction system combining classical astrodynamics and machine learning  
**Author:** Rajrupa Das  
**Status:** Simulation and dataset generation complete; machine learning module in progress

---

## Overview

HOPEX is a hybrid orbit prediction system designed to simulate and forecast the motion of Low Earth Orbit (LEO) satellites under perturbative forces, beginning with the J2 zonal harmonic effect. This project combines traditional orbital propagation with data-driven forecasting approaches, with a focus on maintaining physical fidelity while reducing computational load.

The long-term aim is to evaluate the effectiveness of machine learning models in approximating orbital dynamics and explore techniques to interpret their internal behavior.

---

## Motivation

While traditional orbital propagators like RK4 remain reliable and precise, they can be computationally expensive over long time spans or across many objects. In contrast, machine learning can offer faster approximations, but with less interpretability and physical grounding. This project explores the middle ground: a system that uses real orbital simulations as training data for a machine learning model, guided by physical principles and with transparency in mind.

---

## Project Features

- Orbit propagation using J2 perturbation model via the Poliastro library
- Generation of time-series datasets of position and velocity vectors
- Comparative simulation without J2 for baseline reference
- Structured repository for future integration of ML forecasting and explainability layers

---

## Repository Structure

orbit-prediction-j2-ml/
├── data/ # Raw orbit data (CSV)
│ ├── orbit_j2.csv
│ └── orbit_no_j2.csv
├── src/ # Python source code
│ ├── simulate_orbit.py
│ └── utils.py
├── results/ # Output plots (to be added)
├── research_summary.md # Ongoing technical documentation
├── requirements.txt # Python dependencies
└── README.md # Project overview and setup instructions


