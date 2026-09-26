# Mid-Term Dissertation Progress Report

**Title:** Physics-Informed Neural Networks for Forward and Inverse Problems in Nonlinear Partial Differential Equations  
**Candidate Name:** Priyanshu Kumar  
**Department:** Department of Mathematics  
**Supervisor Review:** Mid-Term Evaluation (Months 1 & 2 Progress)  

---

## 1. Executive Summary & Abstract

Traditional computational fluid dynamics (CFD) and numerical analysis rely extensively on discretization schemes—such as the Finite Difference Method (FDM), Finite Element Method (FEM), and Finite Volume Method (FVM). While robust, these methods suffer from severe drawbacks:
1. High computational burden and the *curse of dimensionality* in high dimensions.
2. Inflexibility when handling irregular geometries or sparse sensor observations.
3. Inability to seamlessly solve **inverse problems** (discovering unknown physical parameters from partial data).

In the first two months of this dissertation, we have investigated **Physics-Informed Neural Networks (PINNs)**—a Scientific Machine Learning (SciML) framework pioneered by Raissi, Perdikaris, and Karniadakis (2017/2019). We transitioned away from legacy implementations (TensorFlow 1.x) to a modern, modular **PyTorch** architecture using `torch.autograd` for exact, mesh-free differential operator evaluation.

We have successfully validated the methodology on two foundational benchmarks:
1. **1D Poisson Boundary Value Problem:** Demonstrated superior accuracy ($L_2 \text{ relative error } \approx 7.0 \times 10^{-4}$) compared to standard 2nd-order Central Difference FDM ($N=50, L_2 \approx 1.37 \times 10^{-3}$).
2. **1D Viscous Burgers' Equation:** Accurately captured non-linear advection, diffusion, and steep shock-front steepening without numerical dispersion or artificial diffusion.

---

## 2. Mathematical Formulation & Framework

Consider a general nonlinear partial differential equation defined on domain $\Omega \subset \mathbb{R}^d$ and time $t \in [0, T]$:

$$\mathcal{D}[u](x, t) := u_t + \mathcal{N}_x[u] = 0, \quad x \in \Omega, \ t \in [0, T]$$

Subject to boundary conditions $\mathcal{B}(u, x, t) = 0$ on $\partial \Omega$ and initial conditions $u(x, 0) = u_0(x)$.

### 2.1 Neural Network Representation
We approximate the latent continuous field $u(x, t)$ with a deep feed-forward neural network parameterized by weights and biases $\theta = \{W^{(l)}, b^{(l)}\}_{l=1}^L$:

$$\hat{u}(x, t; \theta) = \sigma \left( W^{(L)} \dots \sigma(W^{(1)} [x, t]^T + b^{(1)}) \dots + b^{(L)} \right)$$

where $\sigma(\cdot)$ is an infinitely differentiable activation function (e.g., $\tanh$).

### 2.2 Reverse-Mode Automatic Differentiation (Autograd)
Unlike numerical differentiation which introduces truncation error $\mathcal{O}(\Delta x^2)$:
$$\frac{\partial u}{\partial x} \approx \frac{u(x+\Delta x) - u(x-\Delta x)}{2 \Delta x}$$

PINNs compute exact partial derivatives by applying the chain rule directly to the computational graph:
$$\frac{\partial \hat{u}}{\partial x} = \sum_{k} \frac{\partial \hat{u}}{\partial z_k} \frac{\partial z_k}{\partial x}$$

This makes PINNs fundamentally **mesh-free**.

### 2.3 Optimization Objective (Composite Loss)
The parameters $\theta$ are trained by minimizing the composite loss:

$$\mathcal{L}(\theta) = w_{data} \mathcal{L}_{data}(\theta) + w_{pde} \mathcal{L}_{pde}(\theta)$$

$$\mathcal{L}_{data} = \frac{1}{N_u} \sum_{i=1}^{N_u} \left| \hat{u}(x_u^i, t_u^i) - u^i \right|^2$$

$$\mathcal{L}_{pde} = \frac{1}{N_f} \sum_{j=1}^{N_f} \left| \mathcal{D}[\hat{u}](x_f^j, t_f^j) \right|^2$$

where $\{x_f^j, t_f^j\}_{j=1}^{N_f}$ are interior collocation points sampled across the space-time domain.

---

## 3. Computational Experiments & Numerical Results

### Experiment 1: 1D Poisson Boundary Value Problem
$$\frac{d^2 u}{dx^2} + \pi^2 \sin(\pi x) = 0, \quad x \in [-1, 1], \quad u(-1) = u(1) = 0$$
- **Exact Analytical Solution:** $u^*(x) = \sin(\pi x)$
- **Architecture:** 3 hidden layers $\times$ 32 neurons, $\tanh$ activation.
- **Training:** Adam optimizer, 1200 epochs ($\approx 3.2$ seconds runtime on CPU).

#### Quantitative Benchmark:
| Metric | Classical FDM ($N=50$ grid) | PINN (PyTorch Autograd) |
| :--- | :--- | :--- |
| **Relative $L_2$ Error** | $1.3713 \times 10^{-3}$ | **$7.0039 \times 10^{-4}$** |
| **Max Absolute Error** | $1.5204 \times 10^{-3}$ | **$7.4602 \times 10^{-4}$** |
| **Grid Dependency** | Requires structured mesh | **Completely Mesh-free** |

*(Figure reference: `reports/figures/poisson_pinn_vs_fdm.png`)*

---

### Experiment 2: 1D Nonlinear Viscous Burgers' Equation
$$u_t + u u_x - \nu u_{xx} = 0, \quad x \in [-1, 1], \ t \in [0, 1], \quad \nu = \frac{0.01}{\pi}$$
- **Initial Condition:** $u(x, 0) = -\sin(\pi x)$
- **Boundary Condition:** $u(-1, t) = u(1, t) = 0$
- **Significance:** Burgers' equation models shock wave formation where nonlinear convection steepens gradients while diffusion prevents discontinuities.
- **Results:** PINN accurately reproduces the steepening gradient at $x=0$ as $t \to 1.0$ without spurious oscillations.

*(Figure reference: `reports/figures/burgers_pinn_solution.png`)*

---

## 4. Key Milestones Completed (Months 1 & 2)

- [x] Comprehensive review of Raissi et al. (2017) Part I (Solutions) and Part II (Discovery).
- [x] Establishment of clean GitHub version control repository with reproducible architecture.
- [x] PyTorch implementation of automatic differentiation pipeline for differential operators up to 2nd order.
- [x] Classical numerical benchmark implementation (FDM) for direct error comparison.
- [x] Simulation and validation of 1D Poisson problem and 1D Burgers' equation.

---

## 5. Proposed Roadmap for Months 3–6

1. **Month 3 (Inverse Problem & Parameter Discovery):**
   - Formulate parameter estimation framework: Treating viscosity $\nu$ as a learnable parameter.
   - Test parameter recovery under varying noise levels (1%, 5%, 10% Gaussian noise).
2. **Month 4 (Complex Coupled Systems):**
   - Implement the Nonlinear Schrödinger Equation (evaluating complex-valued wave fields).
   - Evaluate continuous-time vs discrete-time (Runge-Kutta) PINNs.
3. **Month 5 (Optimization & Convergence Enhancements):**
   - Implement hybrid Adam $\to$ L-BFGS training routine for machine-precision convergence.
   - Analyze gradient pathology and self-adaptive loss weighting schemes.
4. **Month 6 (Dissertation Drafting & Defense Preparation):**
   - Finalize full LaTeX dissertation document.
   - Prepare final defense presentation slides and viva defense.
