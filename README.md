# Physics-Informed Neural Networks for Forward and Inverse Problems in Nonlinear Partial Differential Equations

**Final Year M.Sc. / B.S. Dissertation Project**  
**Department of Mathematics**  
**Duration:** 6–7 Months (Continuous Research & Development)  
**Author:** Priyanshu Kumar  

---

## 📌 Project Abstract & Overview

Partial Differential Equations (PDEs) are the cornerstone of mathematical modeling in physics, engineering, and biology. Traditional numerical methods (such as the Finite Difference Method (FDM), Finite Element Method (FEM), and Spectral Methods) rely on spatial-temporal discretization (meshing), which often suffers from the curse of dimensionality, complex boundary geometries, and high computational costs.

This dissertation explores **Physics-Informed Neural Networks (PINNs)**—a paradigm in **Scientific Machine Learning (SciML)** that embeds the underlying physical conservation laws (governing differential equations) directly into the loss function of deep neural networks via **Automatic Differentiation (AD)**.

### Primary Objectives:
1. **Theoretical Foundations:** Rigorously study continuous-time and discrete-time PINN architectures for forward and inverse problems.
2. **Modern Re-implementation:** Re-implement foundational benchmarks from Raissi et al. (originally in legacy TensorFlow 1.x) into **modern PyTorch**.
3. **Comparative Numerical Analysis:** Systematically evaluate PINN solutions against classical numerical solvers (FDM/Runge-Kutta) across accuracy ($L_2$ relative error), convergence speed, and grid independence.
4. **Data-Driven Discovery (Inverse Problems):** Discover unknown physical parameters (such as diffusion, viscosity, and reaction coefficients) from noisy and sparse observation data.
5. **Final Deliverables:** A comprehensive academic dissertation report (LaTeX), defense presentation slides, and an extensible SciML software package.

---

## 📂 Repository Structure

```
├── docs/                      # Conceptual guides, math notes & meeting preparation
│   ├── SPRINT_SCHEDULE.md     # 10-day sprint cycle & meeting milestones
│   └── 01_pinn_fundamentals.md# Easy-to-understand guide on PINN mathematics
├── papers/                    # Foundational literature & research papers
│   ├── Raissi_2017_PINNs_Part1_Solutions.pdf
│   └── Raissi_2017_PINNs_Part2_Discovery.pdf
├── src/                       # Production-grade PyTorch implementation
│   ├── models/                # Neural network architectures (MLP, Fourier features)
│   ├── pdes/                  # PDE residual definitions (Burgers, Heat, Schrödinger)
│   └── utils/                 # Data loaders, collocation samplers, error metrics
├── notebooks/                 # Interactive Jupyter notebooks for step-by-step learning
├── reports/                   # Dissertation drafts (LaTeX source, figures, tables)
└── presentations/             # Supervisor update slides & final defense presentation
```

---

## 🔬 Benchmark Problems Under Study

| Equation | Type | Forward Task | Inverse Task |
| :--- | :--- | :--- | :--- |
| **1D Burgers' Equation** | Nonlinear hyperbolic/parabolic | Shock wave propagation | Infer viscosity parameter $\nu$ |
| **1D Heat / Poisson Equation** | Linear parabolic / elliptic | Boundary value verification | Infer thermal diffusivity $\alpha$ |
| **Nonlinear Schrödinger Equation**| Complex dispersive PDE | Soliton wave dynamics | Parameter discovery in dispersion |
| **Incompressible Navier-Stokes** | Coupled nonlinear system | Fluid flow past a cylinder | Infer Reynolds number / pressure field |

---

## 🛠️ Technology Stack
- **Language:** Python 3.10+
- **Deep Learning Framework:** PyTorch (`torch.autograd` for exact gradient computation)
- **Scientific Computing:** NumPy, SciPy (for classical numerical benchmarks)
- **Data Visualization:** Matplotlib, Seaborn
- **Documentation & Typesetting:** LaTeX (Overleaf/TeX Live), Markdown

---

## 📅 Research Supervision Cycle (10-Day Sprints)
Progress is documented and reviewed in structured **10-day sprint cycles** to maintain regular feedback with the faculty supervisor. See [`docs/SPRINT_SCHEDULE.md`](docs/SPRINT_SCHEDULE.md) for the complete roadmap and meeting agendas.
