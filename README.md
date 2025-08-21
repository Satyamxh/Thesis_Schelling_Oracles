# README – Schelling Point Oracle ABM

## Project Overview
This repository contains the source code for the thesis project:  
**“Creating an Agent-Based Model of a Schelling Point Oracle to Analyse Vulnerability to p+ε Attacks.”**

The project implements an **agent-based simulation** (ABM) of a Schelling point oracle, inspired by Kleros, to evaluate how different incentive mechanisms affect resilience against bribery attacks. The simulation is built as a **Streamlit application** for interactivity and visualization.

---

## File Contents

### Core Simulation Files
- **`agents.py`** – Defines juror agents and their decision-making behaviour using Quantal Response Equilibrium (QRE).  
- **`payoff_mechanisms.py`** – Implements different payoff structures (e.g., redistributive vs. non-redistributive incentives).  
- **`model.py`** – Core ABM simulation logic; orchestrates interactions between agents, payoffs, and attacker strategies.  
- **`run.py`** – Main Streamlit entry point; launches the web application interface.  

### Supporting Modules
- **`batch.py`** – Batch simulation page (Streamlit sub-page); allows parameter sweeps across jury size, lambda, ε, etc.  
- **`batch_runner.py`** – Backend functions for running large parameter sweeps in parallel.  
- **`calibration.py`** – Calibration page (Streamlit sub-page); fits lambda and coherence parameter *x* against Kleros General Court data.  
- **`logic.py`** – Helper functions used across the simulation (e.g., payoff calculations, aggregation).  

### Data Extraction
- **`Kleros_json_data_extract.py`** – Parses JSON case data from the Kleros General Court and outputs CSVs of case outcomes for calibration.  

#### Machine Learning

- **`parameter_optimsation.py`** – Optimises model parameters (lambda and x) by fitting against both synthetic simulation results and real Kleros General Court data. Used to calibrate juror rationality and coherence levels.

#### CSV files

- **`/Kleros general court json data/*.csv`** – CSV files generated from real Kleros General Court cases (via `Kleros_json_data_extract.py`).  
- **`batch_results_synthetic_data.csv`** – CSV files produced by the simulation, containing synthetic outcomes for different parameter settings.  
- **`batch_results_attack_symbiotic.csv`**

### Documentation / Setup  

- **`README.md`** – This file.  
- **`CERTIFICATION.txt`** – Authorship certification.  

#### Streamlit Config 
- **/`.streamlit/config.toml`** - removes 200 MB limit for file uploads on Streamlit and sets up custom limit (up to 4GB)
---

## Setup Instructions

### 1. Folder setup

Retain same folder structure (cannot change it)

### 2. Run the App
Launch the Streamlit application:
```bash
streamlit run run.py
```

### 3. Full ABM Experience 
1. `batch.py` can be used to generate data which can be uploaded for analysis.

2. The newly generated `batch_results.csv` can be upladed to generate plots analysing the model data. (It is recommended to vary M, lambda, and x)
