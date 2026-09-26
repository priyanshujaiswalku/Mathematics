# 6–7 Months Research Roadmap & 10-Day Sprint Schedule

**Department of Mathematics — Final Year Dissertation**  
*Supervisor Review Cycle: Every 10 Days*

---

## 🎯 High-Level Phase Overview

| Phase | Duration | Core Focus | Deliverable |
| :--- | :--- | :--- | :--- |
| **Phase 1: Foundations** | Months 1–2 (Sprints 1–6) | Theory, Automatic Differentiation, 1D Toy Problems | Working PyTorch PINN baseline & literature summary |
| **Phase 2: Benchmark PDEs** | Months 2–3 (Sprints 7–10) | Burgers' Eq & Schrödinger Eq (Forward & Inverse) | Numerical validation against FDM benchmarks |
| **Phase 3: Deep Analysis** | Months 3–4 (Sprints 11–13) | Noise sensitivity, loss dynamics (Adam vs L-BFGS) | Comprehensive experimental data & error charts |
| **Phase 4: Novelty & Comparison** | Months 4–5 (Sprints 14–16) | Methodological comparison, hyperparameter analysis | Draft chapters of Dissertation |
| **Phase 5: Writing & Defense** | Months 6–7 (Sprints 17–20) | Final LaTeX thesis report, Beamer/PPT presentation | Complete Thesis PDF & Defense Slides |

---

## 📅 Detailed 10-Day Sprints & Meeting Agendas

### 🔹 Sprint 1 (Days 1 – 10): Foundations & Formal Proposal
- **Tasks:**
  - Read [Raissi Part 1 (Solutions)](../papers/Raissi_2017_PINNs_Part1_Solutions.pdf) Sections 1 & 2.
  - Understand the difference between standard data-driven ML and Physics-Informed ML.
  - Setup Python environment: PyTorch, NumPy, Matplotlib.
- **Supervisor Meeting Agenda:**
  - Present the formalized dissertation title and scope.
  - Discuss the choice of target PDEs (Burgers', Heat, Schrödinger).
  - Show the project roadmap and GitHub repository structure.

---

### 🔹 Sprint 2 (Days 11 – 20): Automatic Differentiation & Toy 1D Problem
- **Tasks:**
  - Understand `torch.autograd.grad` and how exact derivatives without mesh grids are computed.
  - Implement a simple 1D Poisson / 1D Heat Equation PINN in PyTorch.
  - Formulate the total loss: $\mathcal{L}_{total} = \mathcal{L}_{data} + \mathcal{L}_{physics}$.
- **Supervisor Meeting Agenda:**
  - Show live code execution of the 1D Toy Problem.
  - Discuss the convergence of the loss curve.

---

### 🔹 Sprint 3 (Days 21 – 30): 1D Burgers' Equation (Forward Problem)
- **Tasks:**
  - Mathematical formulation of 1D Viscous Burgers' Equation:
    $$u_t + u u_x - \nu u_{xx} = 0, \quad x \in [-1, 1], \ t \in [0, 1]$$
  - Sample collocation points in space-time ($x, t$) using Latin Hypercube Sampling (LHS).
  - Train PINN to approximate $u(x, t)$ with known $\nu = \frac{0.01}{\pi}$.
- **Supervisor Meeting Agenda:**
  - Show the 2D contour plot of $u(x, t)$ predicting shock formation.
  - Present relative $L_2$ error metric against the analytical/numerical ground truth.

---

### 🔹 Sprint 4 (Days 31 – 40): Classical Numerical Solver (FDM Benchmark)
- **Tasks:**
  - Implement a classical Finite Difference Method (FDM / Crank-Nicolson or Runge-Kutta) for the Burgers' Equation.
  - Compare computational cost: Training time vs Grid step size ($\Delta x, \Delta t$) stability constraints (CFL condition).
- **Supervisor Meeting Agenda:**
  - Demonstrate: *Why PINNs are mesh-free* and how they compare with FDM.
  - Review FDM vs PINN error comparison table.

---

### 🔹 Sprint 5 (Days 41 – 50): Inverse Problem (Data-Driven Parameter Discovery)
- **Tasks:**
  - Read [Raissi Part 2 (Discovery)](../papers/Raissi_2017_PINNs_Part2_Discovery.pdf).
  - Treat viscosity $\nu$ as a learnable parameter `torch.nn.Parameter`.
  - From sparse observations of $u(x, t)$, let the neural network discover the true value of $\nu$.
- **Supervisor Meeting Agenda:**
  - Present parameter convergence history (e.g., initial guess $\nu_0 = 0.0$ converging to $\frac{0.01}{\pi} \approx 0.00318$).
  - Discuss the robustness of parameter identification.

---

### 🔹 Sprint 6 (Days 51 – 60): Discrete-Time PINN & Runge-Kutta Formulation
- **Tasks:**
  - Understand the discrete-time formulation from Raissi Part 1 (implicit Runge-Kutta schemes with arbitrary stages $q$).
  - Implement a multi-step RK-PINN to advance solution through large time steps.
- **Supervisor Meeting Agenda:**
  - Compare continuous-time PINN vs discrete-time RK-PINN.
  - Complete Mid-Term Progress Report for the department.

---

### 🔹 Sprint 7 (Days 61 – 70): Nonlinear Schrödinger Equation (Complex-Valued PDE)
- **Tasks:**
  - Formulate complex-valued PDE:
    $$i h_t + 0.5 h_{xx} + |h|^2 h = 0$$
    Split into real and imaginary components: $u(x, t)$ and $v(x, t)$.
  - Train dual-output PINN predicting $[u, v]$ and calculate $|h| = \sqrt{u^2 + v^2}$.
- **Supervisor Meeting Agenda:**
  - Present periodic boundary condition handling in PINNs.
  - Show soliton dynamics plots.

---

### 🔹 Sprint 8 (Days 71 – 80): Noise Sensitivity & Robustness Analysis
- **Tasks:**
  - Add Gaussian noise (1%, 5%, 10%) to measurement data:
    $$u_{noisy} = u + \epsilon \cdot \mathcal{N}(0, \sigma^2)$$
  - Measure how error in discovered parameters degrades with noise.
- **Supervisor Meeting Agenda:**
  - Discuss practical applicability: How real-world noisy sensor data is handled by PINNs without overfitting.

---

### 🔹 Sprint 9 (Days 81 – 90): Optimization Dynamics (Adam vs L-BFGS Hybrid)
- **Tasks:**
  - Analyze why Adam optimizer alone plateaus early in PDE residuals.
  - Implement a 2-stage training pipeline: First Adam (for rough global convergence), then L-BFGS (quasi-Newton method for second-order precision).
- **Supervisor Meeting Agenda:**
  - Compare loss convergence plots between pure Adam vs Adam + L-BFGS.

---

### 🔹 Sprint 10 – 12 (Months 4): Advanced Extensions & Custom Contributions
- **Tasks:**
  - Investigate self-adaptive loss weights (soft-attention / gradient pathology resolution).
  - Test activation functions (Tanh vs GELU vs Sinusoidal/SIREN).
- **Supervisor Meeting Agenda:**
  - Finalize the unique/novel contribution for the dissertation.

---

### 🔹 Sprint 13 – 16 (Month 5): Full Numerical Synthesis & Chapter Drafting
- **Tasks:**
  - Structure the Dissertation in LaTeX:
    - Chapter 1: Introduction & Literature Survey
    - Chapter 2: Mathematical Foundations of PDEs & Neural Networks
    - Chapter 3: Continuous & Discrete PINN Methodology
    - Chapter 4: Forward Benchmark Solutions & FDM Comparison
    - Chapter 5: Inverse Parameter Discovery & Noise Robustness
    - Chapter 6: Conclusion & Future Outlook
- **Supervisor Meeting Agenda:**
  - Submit Draft Chapters 1, 2, and 3 for supervisor review and feedback.

---

### 🔹 Sprint 17 – 20 (Months 6 – 7): Final Review, Presentation (PPT) & Defense
- **Tasks:**
  - Revise dissertation based on supervisor's feedback.
  - Prepare high-impact Beamer / PowerPoint presentation (slides with animations, equations, and result figures).
  - Mock defense viva Q&A.
- **Supervisor Meeting Agenda:**
  - Final sign-off on the Dissertation document.
  - Final Defense Presentation rehearsal.
