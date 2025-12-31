# Heston Model Calibration Using Numerical Optimisation

This repository presents a **production-grade, fully reproducible calibration framework for the Heston stochastic volatility model**, with explicit focus on **numerical stability, optimisation design, and diagnostic validation**.

*For questions or discussion related to this work, feel free to contact **Muhammad Shoaib** at **safridi@gmail.com**.*

It accompanies the technical research note:

> **Heston Model Calibration Using Numerical Optimisation**  
> Dr. Muhammad Shoaib  
> *(PDF included in this repository)*

This project is designed as a **quantitative research and implementation showcase**, suitable for reviewers in:
- Quantitative Research & Model Development  
- Derivatives Pricing & Volatility Modelling  
- Model Validation / Independent Model Review  
- Risk & Front-Office Quant Engineering  

---

## What This Repository Demonstrates

This project goes beyond formula transcription. It demonstrates how **Heston calibration behaves as a numerical optimisation problem in practice**, including its structural limitations.

Specifically, it provides:

- A **clean, numerically stable implementation** of the Heston characteristic function  
- Fourier-based European option pricing using **Lewis (2001)** inversion  
- **Constrained nonlinear least-squares calibration** via Levenberg–Marquardt  
- Explicit handling of:
  - complex-valued arithmetic and branch selection  
  - parameter constraints and non-convexity  
  - solver sensitivity and local minima  
- **Surface-level diagnostics**:
  - implied volatility smiles  
  - full maturity–strike surfaces  
  - residual heatmaps  
- **Fully reproducible synthetic calibration experiments** isolating numerical effects from market noise  

The emphasis is on **robust numerical methodology**, not black-box optimisation or one-off fits.

---

## Key Takeaways for Practitioners

**Surface fit matters more than parameter recovery.**  
Even in a noise-free synthetic setting, multiple Heston parameter vectors generate virtually indistinguishable implied volatility surfaces. This non-uniqueness is a structural property of the model, not a calibration failure.

**Numerical stability is as important as closed-form pricing.**  
Correct pricing formulas alone are insufficient. Practical calibration requires careful control of branch cuts, characteristic function representations (e.g. Little Trap), and solver behaviour to avoid silent numerical failure.

**Multi-start optimisation is essential.**  
The calibration objective is non-convex and exhibits flat directions. Robust pipelines must use multi-start strategies and judge success via surface diagnostics rather than a single “optimal” parameter vector.

**Diagnostics should be surface-based, not parameter-based.**  
Residual heatmaps across strike and maturity provide a more reliable assessment of calibration quality than parameter values, especially when parameters are weakly identified.

**Synthetic experiments are a powerful validation tool.**  
Calibrating to data with known ground truth separates numerical behaviour from market microstructure effects and provides a controlled environment for testing optimisation stability.

---

## Scope

This repository focuses on **numerical calibration methodology** rather than live market data ingestion, bid–ask modelling, or desk-specific data pipelines. The framework is intentionally designed to be **transparent, reproducible, and extensible** for research, validation, and comparative model studies.

# Professional Relevance

This repository demonstrates **hands-on experience with production-style quantitative modelling**, beyond theoretical derivations.

Specifically, it evidences:
- End-to-end ownership of a **nonlinear calibration pipeline** for a stochastic volatility model  
- Practical handling of **numerical stability, solver design, and diagnostics**  
- Awareness of **model risk issues**, including parameter non-uniqueness and surface-level validation  
- Ability to translate mathematical models into **reproducible, inspectable Python code**  

This work is representative of tasks encountered in **quantitative research, derivatives pricing, and model validation roles**, including model implementation, calibration, and diagnostic review.

For further technical detail, see the accompanying PDF and reproducible numerical experiments in this repository.



## Repository structure

heston_calibration/
│
├── heston/ # Core model and calibration code
│ ├── cf.py # Heston characteristic function
│ ├── pricing.py # Fourier-based pricing (Lewis)
│ ├── implied_vol.py # Black–Scholes IV inversion
│ ├── calibration.py # Levenberg–Marquardt optimiser
│ └── utils.py
│
├── examples/ # Reproducible experiments
│ ├── synthetic_smile_experiment.py
│ ├── report_tables.py
│ ├── run_calibration.py
│ └── utils_metrics.py
│
├── tables/ # Auto-generated LaTeX tables
│ ├── table_true_vs_cal.tex
│ ├── table_metrics.tex
│ └── table_multistart.tex
│
├── fig_1_heston_smile_T1Y.png
├── fig_2_market_vs_model_iv_surface.png
├── fig_3_residual_heatmap.png
│
├── Heston Model Calibration Using Numerical Optimisation.pdf
└── README.md


---

## Numerical Experiments (Highlights)

All experiments are performed on **synthetic data generated from known Heston parameters**, allowing controlled validation of the calibration pipeline independent of market noise.

---

### 1. Implied Volatility Smile Fit

The calibrated model reproduces the synthetic implied volatility smile at \(T = 1\) year **to numerical precision**.

![Smile Fit](fig_1_heston_smile_T1Y.png)

---



### 2. Full Surface Comparison

Comparison between the **synthetic implied volatility surface** (left) and the **surface implied by the calibrated Heston model** (right).

Despite the well-known **non-uniqueness of Heston parameters**, surface-level agreement is excellent across strikes and maturities.

![Surface Fit](fig_2_market_vs_model_iv_surface.png)

---

### 3. Residual Diagnostics

Residual implied volatilities  
\[
\widehat{\sigma}_{\text{model}} - \sigma_{\text{synthetic}}
\]
are small, stable, and exhibit **no systematic moneyness- or maturity-dependent bias**.

![Residual Heatmap](fig_3_residual_heatmap.png)

---

## Quantitative Fit Metrics

Calibration accuracy is summarised using implied-volatility errors:

- **RMSE**
- **MAE**
- **Maximum absolute error**

Tables are stored in the tables/ directory and included in the accompanying paper.

## Quick Start (Reproducibility)

To reproduce the numerical experiments and figures:

```bash
pip install numpy scipy matplotlib
python examples/run_calibration.py
```

## Contact

**Muhammad Shoaib**  
Email: safridi@gmail.com   

