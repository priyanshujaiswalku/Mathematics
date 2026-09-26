# Mathematical Foundations of Physics-Informed Neural Networks (PINNs)

**Department of Mathematics — Advanced Theoretical & Computational Compendium**  
**Candidate:** Priyanshu Kumar  
**Specialization:** Applied Mathematics, Scientific Machine Learning (SciML) & Numerical PDE Analysis  

---

## 🏛️ 1. Abstract Operator Formulation of PDEs

In functional analysis, a general non-linear partial differential equation can be formulated as an operator equation on appropriate function spaces.

Let $\Omega \subset \mathbb{R}^d$ be a bounded open domain with Lipschitz boundary $\partial \Omega$, and let $[0, T]$ denote the temporal domain. We denote the spatiotemporal cylinder as:
$$Q_T = \Omega \times (0, T], \quad \Sigma_T = \partial \Omega \times (0, T]$$

We consider the general Cauchy-Dirichlet Initial-Boundary Value Problem (IBVP):

$$\begin{aligned}
\mathcal{P}[u](x, t) &:= \frac{\partial u}{\partial t} + \mathcal{N}_x[u] - g(x, t) = 0, \quad &&(x, t) \in Q_T \\
\mathcal{B}[u](x, t) &:= u(x, t) - h(x, t) = 0, \quad &&(x, t) \in \Sigma_T \\
\mathcal{I}[u](x) &:= u(x, 0) - u_0(x) = 0, \quad &&x \in \Omega
\end{aligned}$$

Where:
- $\mathcal{N}_x: \mathcal{V} \to \mathcal{V}^*$ is a non-linear spatial differential operator acting on a Sobolev space $\mathcal{V} \subseteq H^k(\Omega)$.
- $\mathcal{B}$ denotes the trace/boundary operator.
- $\mathcal{I}$ denotes the Cauchy initial data operator.
- $g \in L^2(Q_T)$, $h \in L^2(\Sigma_T)$, and $u_0 \in L^2(\Omega)$ are known source and boundary data.

---

## 🧠 2. Deep Neural Networks as Universal Approximators

### 2.1 The Multi-Layer Perceptron (MLP) Representation
Let the neural network approximation be denoted by $\hat{u}(x, t; \theta) \in C^k(Q_T)$, parameterized by weights and biases $\theta \in \mathbb{R}^P$:

$$\hat{u}(x, t; \theta) = \mathcal{W}_L \circ \sigma \circ \mathcal{W}_{L-1} \circ \dots \circ \sigma \circ \mathcal{W}_1(z)$$

Where $z = [x_1, \dots, x_d, t]^T \in \mathbb{R}^{d+1}$ is the input coordinate vector, and:
$$\mathcal{W}_l(a) = W^{(l)} a + b^{(l)}, \quad W^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}, \quad b^{(l)} \in \mathbb{R}^{d_l}$$

### 2.2 Theorem: Universal Approximation of Derivatives (Pinkus, 1999; Hornik et al., 1990)
> **Theorem (Simultaneous Approximation of Functions and Derivatives):**  
> Let $\sigma \in C^m(\mathbb{R})$ be a non-polynomial activation function. Then, for any target function $u \in C^m(K)$ on a compact set $K \subset \mathbb{R}^{d+1}$ and any $\epsilon > 0$, there exists an MLP $\hat{u}(z; \theta)$ such that:
> $$\max_{|\alpha| \le m} \sup_{z \in K} \left| D^\alpha u(z) - D^\alpha \hat{u}(z; \theta) \right| < \epsilon$$
> where $D^\alpha = \frac{\partial^{|\alpha|}}{\partial z_1^{\alpha_1} \dots \partial z_{d+1}^{\alpha_{d+1}}}$ is a multi-index partial derivative.

### 2.3 Mathematical Significance of Activation Function Smoothness:
- Standard Deep Learning uses **$\text{ReLU}(z) = \max(0, z)$**. However, $\text{ReLU} \notin C^2(\mathbb{R})$.
  - $\frac{d}{dz} \text{ReLU}(z) = \text{Heaviside}(z)$
  - $\frac{d^2}{dz^2} \text{ReLU}(z) = \delta(z) = 0 \quad (\text{almost everywhere for } z \neq 0)$
- **Conclusion for Mathematics:** For 2nd-order PDEs (e.g. Laplacian $\nabla^2 u$, Diffusion $\nu u_{xx}$), $\text{ReLU}$ causes the PDE residual to vanish to zero almost everywhere, making physics training impossible!
- **Requirement:** We must choose smooth, infinitely differentiable activations: $\sigma(z) = \tanh(z) \in C^\infty(\mathbb{R})$ or $\sigma(z) = \sin(z)$ (SIREN).

---

## ⚡ 3. Reverse-Mode Automatic Differentiation (Autograd)

To evaluate the PDE differential operator $\mathcal{P}[\hat{u}]$, we require exact partial derivatives without spatial discretization.

### 3.1 Taylor Series Truncation Error in Classical FDM:
$$\frac{u(x+h) - u(x-h)}{2h} = u'(x) + \underbrace{\frac{h^2}{6} u'''(\xi)}_{\mathcal{O}(h^2) \text{ Discretization Error}}$$
As $h \to 0$, round-off error from finite machine precision ($\epsilon_{mach}$) dominates: $\text{Error} \sim \mathcal{O}(h^2) + \frac{\epsilon_{mach}}{h}$.

### 3.2 Exact Derivative via Computational Graph & Chain Rule:
Neural network evaluation constructs a directed acyclic graph (DAG) of elementary operations:
$$v_i = \phi_i\left(\text{Parents}(v_i)\right), \quad i = 1, \dots, M$$
By applying the reverse-mode chain rule (Adjoint State / Vector-Jacobian Product):
$$\bar{v}_j = \frac{\partial \hat{u}}{\partial v_j} = \sum_{k \in \text{Children}(v_j)} \bar{v}_k \frac{\partial v_k}{\partial v_j}$$
- **Exactness:** The derivative $\frac{\partial \hat{u}}{\partial x_i}$ is calculated to **machine precision** without any grid truncation error ($\mathcal{O}(h^2) = 0$).

---

## 🎯 4. Continuous Variational Loss Functional & Monte Carlo Quadrature

We define the theoretical continuous objective functional in the $L^2$-Hilbert space norm:

$$\mathcal{J}(\theta) = \lambda_f \|\mathcal{P}[\hat{u}]\|_{L^2(Q_T)}^2 + \lambda_b \|\mathcal{B}[\hat{u}]\|_{L^2(\Sigma_T)}^2 + \lambda_0 \|\mathcal{I}[\hat{u}]\|_{L^2(\Omega)}^2$$

where the continuous inner-product norm is defined as:
$$\|\mathcal{P}[\hat{u}]\|_{L^2(Q_T)}^2 = \int_{0}^T \int_{\Omega} \left| \frac{\partial \hat{u}}{\partial t} + \mathcal{N}_x[\hat{u}] - g(x, t) \right|^2 dx \, dt$$

### Monte Carlo Quadrature Discretization:
Evaluating high-dimensional integrals via numerical Riemann sums suffers from exponential complexity $\mathcal{O}(N^d)$. By sampling $N_f$ interior collocation points $\{x_f^i, t_f^i\}_{i=1}^{N_f} \sim \text{Uniform}(Q_T)$, Monte Carlo integration yields the empirical loss:

$$\|\mathcal{P}[\hat{u}]\|_{L^2(Q_T)}^2 \approx \frac{\text{Vol}(Q_T)}{N_f} \sum_{i=1}^{N_f} \left| \mathcal{P}[\hat{u}](x_f^i, t_f^i) \right|^2$$

Hence, the total empirical loss optimized via Gradient Descent / L-BFGS is:

$$\mathcal{L}(\theta) = \frac{\lambda_f}{N_f} \sum_{i=1}^{N_f} \left| \mathcal{P}[\hat{u}](x_f^i, t_f^i) \right|^2 + \frac{\lambda_b}{N_b} \sum_{j=1}^{N_b} \left| \hat{u}(x_b^j, t_b^j) - h(x_b^j, t_b^j) \right|^2 + \frac{\lambda_0}{N_0} \sum_{k=1}^{N_0} \left| \hat{u}(x_0^k, 0) - u_0(x_0^k) \right|^2$$

---

## 🔬 5. Mathematical Equations & Physical Applications

```
                                 ┌──────────────────────────────────────────────┐
                                 │       Nonlinear Evolution PDEs               │
                                 └──────────────────────┬───────────────────────┘
                                                        │
                      ┌─────────────────────────────────┴─────────────────────────────────┐
                      ▼                                                                   ▼
       ┌──────────────────────────────┐                                    ┌──────────────────────────────┐
       │   Burgers' Equation (Fluid)  │                                    │ Schrödinger Equation (Optics)│
       │   u_t + u*u_x - ν*u_xx = 0   │                                    │   i*h_t + 0.5*h_xx + |h|²h=0 │
       └──────────────────────────────┘                                    └──────────────────────────────┘
```

### 5.1 The 1D Poisson Equation (Elliptic BVP)
$$-\nabla^2 u(x) = f(x) \iff -\frac{d^2 u}{dx^2} = \pi^2 \sin(\pi x), \quad x \in [-1, 1]$$
- **Boundary Conditions:** $u(-1) = 0, \quad u(1) = 0$
- **Exact Analytical Solution:** $u^*(x) = \sin(\pi x)$
- **Physical Applications:**
  1. **Electrostatics:** Gauss's Law in differential form ($\nabla^2 \phi = -\rho / \epsilon_0$).
  2. **Steady-State Heat Transfer:** Temperature distribution with internal heat generation.
  3. **Gravitation:** Newtonian gravitational potential ($\nabla^2 \Phi = 4 \pi G \rho$).

---

### 5.2 The 1D Viscous Burgers' Equation (Nonlinear Parabolic PDE)
$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0, \quad x \in [-1, 1], \ t \in [0, 1]$$
Where $\nu = \frac{0.01}{\pi}$ is the kinematic viscosity.
- **Decomposition of Terms:**
  - $\frac{\partial u}{\partial t}$: Unsteady rate of change.
  - $u \frac{\partial u}{\partial x}$: **Nonlinear advection/convection** (causes wave steepening and shock formation).
  - $-\nu \frac{\partial^2 u}{\partial x^2}$: **Linear diffusion/dissipation** (smoothes discontinuities).
- **Physical Applications:**
  1. **Simplified Navier-Stokes:** Prototype 1D model for hydrodynamic turbulence and boundary layer analysis.
  2. **Acoustic Shock Waves:** Propagation of finite-amplitude sound waves in dissipative media.
  3. **Traffic Flow Dynamics:** Lighthill-Whitham-Richards model for vehicular shock waves on highways.

---

### 5.3 The Nonlinear Schrödinger Equation (NLSE) (Dispersive Complex PDE)
$$i \frac{\partial h}{\partial t} + \frac{1}{2} \frac{\partial^2 h}{\partial x^2} + |h|^2 h = 0$$
Where $h(x, t) = u(x, t) + i v(x, t) \in \mathbb{C}$.
- **Coupled System for PINN:**
  $$\begin{cases}
  \frac{\partial u}{\partial t} + \frac{1}{2} \frac{\partial^2 v}{\partial x^2} + (u^2 + v^2) v = 0 \\
  \frac{\partial v}{\partial t} - \frac{1}{2} \frac{\partial^2 u}{\partial x^2} - (u^2 + v^2) u = 0
  \end{cases}$$
- **Physical Applications:**
  1. **Quantum Mechanics:** Bose-Einstein condensates described by the Gross-Pitaevskii equation.
  2. **Fiber Optics:** Optical solitons propagating through nonlinear optical fibers without dispersion.
  3. **Deep Water Waves:** Rogue waves and modulation instability in oceanography.

---

## 📊 6. Rigorous Error Metrics & Sobolev Norms

In mathematical evaluation, we assess the approximation error $e(x) = \hat{u}(x) - u^*(x)$ using formal function space norms:

1. **Relative $L^2$ Error (Root-Mean-Square Metric):**
   $$\mathcal{E}_{L^2} = \frac{\|\hat{u} - u^*\|_{L^2(\Omega)}}{\|u^*\|_{L^2(\Omega)}} = \frac{\sqrt{\int_\Omega |\hat{u}(x) - u^*(x)|^2 dx}}{\sqrt{\int_\Omega |u^*(x)|^2 dx}} \approx \frac{\sqrt{\sum_{i=1}^{M} |\hat{u}(x_i) - u^*(x_i)|^2}}{\sqrt{\sum_{i=1}^{M} |u^*(x_i)|^2}}$$

2. **Chebyshev / Uniform ($L^\infty$) Error (Worst-Case Pointwise Bound):**
   $$\mathcal{E}_{L^\infty} = \|\hat{u} - u^*\|_{L^\infty(\Omega)} = \text{ess sup}_{x \in \Omega} |\hat{u}(x) - u^*(x)| \approx \max_{1 \le i \le M} |\hat{u}(x_i) - u^*(x_i)|$$

3. **Sobolev $H^1$ Semi-Norm Error (Gradient Accuracy):**
   $$|e|_{H^1(\Omega)} = \left( \int_\Omega |\nabla \hat{u}(x) - \nabla u^*(x)|^2 dx \right)^{1/2}$$
   *(Ensures the neural network not only matches the function values, but also correctly predicts physical flux and forces!)*

---

## 🔄 7. PINN Computational Architecture Diagram

```mermaid
flowchart TD
    subgraph Inputs["1. Space-Time Coordinates"]
        coords["(x, t) ~ Q_T"]
    end

    subgraph NeuralNet["2. Deep Neural Approximator"]
        coords --> W1["Affine Layer: W¹·z + b¹"]
        W1 --> Act1["Activation: tanh(·)"]
        Act1 --> Hidden["Hidden Layers (Depth L, Width W)"]
        Hidden --> ActL["Activation: tanh(·)"]
        ActL --> Output["Predicted Field: u_hat(x, t; θ)"]
    end

    subgraph AutogradEngine["3. PyTorch Autograd Engine (Exact Chain Rule)"]
        Output -->|du/dx| G1["Spatial Gradient: u_x"]
        Output -->|du/dt| G2["Temporal Gradient: u_t"]
        G1 -->|d²u/dx²| G3["Laplacian / Diffusion: u_xx"]
    end

    subgraph PhysicsResidual["4. Differential Operator Evaluation"]
        G1 & G2 & G3 --> Residual["PDE Residual: f = u_t + u·u_x - ν·u_xx"]
    end

    subgraph Objective["5. Multi-Objective Optimization"]
        Residual --> LossPDE["L_pde = (1/N_f) Σ |f|²"]
        Output --> LossBC["L_data = (1/N_u) Σ |u_hat - u_true|²"]
        LossPDE & LossBC --> TotalLoss["Total Loss: L(θ) = λ_u·L_data + λ_f·L_pde"]
    end

    subgraph Backpropagation["6. Gradient Update"]
        TotalLoss --> Optimizer["Optimizer: Adam → L-BFGS"]
        Optimizer -->|Update θ = {W, b}| NeuralNet
    end
```
