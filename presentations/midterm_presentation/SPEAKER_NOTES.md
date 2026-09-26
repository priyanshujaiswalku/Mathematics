# Mid-Term Presentation: Slide-by-Slide Speaker Notes & Viva Script

**Candidate:** Priyanshu Kumar  
**Presentation:** `presentations/midterm_presentation/Midterm_PINNs_Presentation.pptx`  
**Purpose:** Use this script to speak confidently during your mid-term review in front of your supervisor and departmental committee.

---

### 🎙️ Slide 1: Title Slide
* **English:** *"Respected Supervisor and Committee Members, Good morning. Today, I am presenting my mid-term dissertation progress on the topic: 'Physics-Informed Neural Networks for Forward and Inverse Problems in Nonlinear Partial Differential Equations'."*
* **Hindi Samajh:** Sabse pehle apna naam, department aur dissertation ka title bolna hai.

---

### 🎙️ Slide 2: Research Motivation & Problem Statement
* **English:** *"To set the context: In computational mathematics, solving PDEs has traditionally relied on mesh-based methods like FDM and FEM. However, mesh generation in higher dimensions faces the curse of dimensionality and fails when dealing with sparse or noisy observations.  
Our objective is to leverage Physics-Informed Neural Networks (PINNs), where physical conservation laws are embedded directly into the neural network via Automatic Differentiation, providing a truly mesh-free alternative."*
* **Hindi Samajh:** Yahan explain karna hai ki classical methods (FDM/FEM) me mesh banana kitna mushkil hota hai aur PINN kaise mesh-free hai.

---

### 🎙️ Slide 3: Mathematical Framework & Composite Loss
* **English:** *"Here is our mathematical formulation. We frame the PDE as an operator equation in Hilbert Space $L^2(Q_T)$. According to Pinkus' Universal Approximation Theorem for Derivatives, an MLP with $C^\infty$ activation functions (like $\tanh$) can approximate both the function and its partial derivatives simultaneously.  
Our loss functional discretizes the continuous $L^2$ norm via Monte Carlo quadrature into $\mathcal{L}_{data}$ and $\mathcal{L}_{pde}$."*
* **Hindi Samajh:** Hilbert space, Pinkus Theorem for Derivatives, aur Monte Carlo integration ka formal mathematical reference dijiye.

---

### 🎙️ Slide 4 (Slide 3B): Computational Architecture & Workflow Diagram
* **English:** *"This architectural diagram illustrates our complete computational pipeline:
1. Spatiotemporal coordinates $(x, t)$ enter the deep network approximator.
2. The network outputs predicted field $\hat{u}(x, t)$.
3. Crucially, PyTorch's Autograd engine computes exact partial derivatives ($\hat{u}_t, \hat{u}_x, \hat{u}_{xx}$) via the chain rule on the computational graph.
4. These gradients feed into the PDE residual operator $f(x, t) = \hat{u}_t + \hat{u}\hat{u}_x - \nu \hat{u}_{xx}$.
5. The composite loss is minimized via backpropagation to iteratively optimize weights $\theta$."*
* **Hindi Samajh:** Screen par bane diagram ko step-by-step point kijiye: Coordinates $\to$ Neural Net $\to$ Autograd $\to$ Residual $\to$ Loss $\to$ Backpropagation update loop. Committee diagram dekh kar bohot impress hogi!

---

### 🎙️ Slide 4 & 5: Benchmark 1: 1D Poisson Equation (PINN vs FDM)
* **English:** *"To validate our implementation, we solved a 1D Poisson boundary value problem: $-u''(x) = \pi^2 \sin(\pi x)$ with zero boundary conditions, which has an exact analytical solution $u^*(x) = \sin(\pi x)$.  
As shown in the figures and comparison table:
- Classical FDM with 50 grid points yielded a relative $L_2$ error of $1.37 \times 10^{-3}$.
- Our PyTorch PINN achieved a relative $L_2$ error of $7.00 \times 10^{-4}$—which is nearly 49% lower error, while requiring zero spatial mesh discretization."*
* **Hindi Samajh:** Graph aur table dikhayein! Bohein ki humne FDM aur PINN dono ko run kiya, aur PINN ne 49% kam error ke saath solve kiya bina kisi mesh ke.

---

### 🎙️ Slide 6: Benchmark 2: 1D Viscous Burgers' Equation
* **English:** *"Next, we tackled the nonlinear Viscous Burgers' Equation: $u_t + u u_x - \nu u_{xx} = 0$, with $\nu = 0.01/\pi$. This is a classical nonlinear benchmark modeling fluid advection and diffusion.  
As seen in the 2D contour plot and time snapshots, as time progresses from $t=0$ to $t=1$, the wave profile steepens, successfully forming a sharp internal gradient (shock front) at $x=0$ without numerical instability or artificial diffusion."*
* **Hindi Samajh:** 2D heatmap aur time-slice plot dikhayein. Batayein ki kaise $t=0$ par sine wave thi aur $t=1$ tak aate aate shock front ban gaya, jisko neural network ne smoothly capture kiya.

---

### 🎙️ Slide 7: Mid-Term Achievements (Months 1 & 2)
* **English:** *"To summarize our work in the first two months:
1. Conducted an extensive literature review of the seminal papers by Raissi et al. (2017).
2. Built a modern, modular PyTorch codebase from scratch, avoiding obsolete TensorFlow 1.x implementations.
3. Implemented and validated two key benchmarks: 1D Poisson and 1D Burgers' equations.
4. Set up an open, reproducible GitHub research repository and documentation."*
* **Hindi Samajh:** Teacher ko clear message milega ki 2 mahine me aapne literature review, modern code implementation, aur 2 working models complete kar liye hain.

---

### 🎙️ Slide 8: Roadmap for Remaining Months (3 to 6)
* **English:** *"Looking ahead to the next phase:
- In Month 3, we will implement Inverse Problems: discovering unknown viscosity $\nu$ from sparse and noisy data.
- In Month 4, we will examine the Nonlinear Schrödinger equation (complex-valued wave dynamics).
- In Month 5, we will evaluate hybrid Adam to L-BFGS optimization and adaptive loss weighting.
- In Month 6, we will finalize the complete dissertation report and defense presentation."*
* **Hindi Samajh:** Aage ka poora roadmap unke saamne rakhein taaki committee ko pata ho ki aapka har step planned hai.

---

### 🎙️ Slide 9: Conclusion & Viva Q&A
* **English:** *"Thank you for your time and attention. I am now open to your questions, feedback, and guidance."*
