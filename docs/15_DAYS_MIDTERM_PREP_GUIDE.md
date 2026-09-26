# 15-Day Mid-Term Presentation Preparation Guide

**Candidate:** Priyanshu Kumar  
**Focus:** Pure Mid-Term Mastery (No Future Overload)  
**Target:** Ace the Mid-Term Evaluation in 15 Days  

---

## 🎯 Plan Overview: 3 Simple 5-Day Phases

```
[Phase 1: Days 1 to 5]     -->     [Phase 2: Days 6 to 10]     -->     [Phase 3: Days 11 to 15]
  Concepts & Math Clarity           Code & Graph Hands-on             Rehearsal & Viva Defense
```

---

## 📘 Phase 1: Mathematical Concepts (Days 1 – 5)
*Goal: Understand the core concepts so you never get stuck on basic theory.*

### Day 1: The Problem with Traditional Methods
- **Kya samajhna hai:** Classical methods (FDM/FEM) me mesh (grid) banana padta hai.
- **Kyu mushkil hai:** Agar geometry complex ho ya dimensions zyada hon, toh mesh banana bahut slow aur computationally heavy hota hai.
- **Key Term:** *Curse of Dimensionality*.

### Day 2: What is a PINN?
- **Kya samajhna hai:** PINN ek aam Neural Network hi hai, bas iske loss function me Physics (PDE) ka formula juda hota hai.
- **Loss Equation:**
  $$\mathcal{L} = \mathcal{L}_{data} + \mathcal{L}_{pde}$$
- $\mathcal{L}_{data}$: Boundary points par kitna sahi answer de raha hai.
- $\mathcal{L}_{pde}$: Domain ke andar formula kitna sahi follow kar raha hai.

### Day 3: Automatic Differentiation (PyTorch Autograd)
- **Teacher ka favourite question:** *"Derivatives kaise nikalte ho?"*
- **Answer:** Finite Difference formula $(u(x+\Delta x)-u(x))/\Delta x$ se nahi! Hum PyTorch ke `torch.autograd` se chain rule lagakar exact analytical derivative nikalte hain.
- **Benefit:** Truncation error zero hota hai aur mesh ki zaroorat nahi hoti.

### Day 4: Benchmark 1 - 1D Poisson Equation
- Equation: $-u''(x) = \pi^2 \sin(\pi x)$ on $[-1, 1]$.
- Exact Solution: $u^*(x) = \sin(\pi x)$.
- Kyu chuna: Iska exact answer pata hai, isliye hum PINN aur FDM dono ka error exact calculate kar sakte hain.

### Day 5: Benchmark 2 - 1D Burgers' Equation
- Equation: $u_t + u u_x - \nu u_{xx} = 0$.
- Kyu important hai: Yeh fluid mechanics ka equation hai. Isme wave chalte-chalte steep ho jaati hai (shock wave banti hai).

---

## 💻 Phase 2: Code & Graph Familiarity (Days 6 – 10)
*Goal: Understand the results generated on your computer.*

### Day 6 & 7: 1D Poisson Code (`src/poisson_1d_pinn.py`)
- Code ko terminal me run karke dekhein: `python src/poisson_1d_pinn.py`
- Notice karein: 3 second me train ho jata hai.
- Numbers yaad rakhein:
  - FDM Error: $1.37 \times 10^{-3}$
  - PINN Error: $7.00 \times 10^{-4}$ (PINN ne 49% kam error diya!).

### Day 8 & 9: 1D Burgers' Code (`src/burgers_1d_pinn.py`)
- Code ko run karke dekhein: `python src/burgers_1d_pinn.py`
- Plot open karein: `reports/figures/burgers_pinn_solution.png`
- Dekhein ki kaise $t=0$ par sine curve tha aur $t=1$ par shock wave ban gayi.

### Day 10: Reviewing the Presentation Slides
- Open karein: `presentations/midterm_presentation/Midterm_PINNs_Presentation.pptx`
- Har slide ke layout aur content ko ek baar dekh lijiye.

---

## 🎤 Phase 3: Presentation Rehearsal & Viva Defense (Days 11 – 15)
*Goal: Speak fluently and handle any cross-questioning.*

### Day 11 & 12: Slide-by-Slide Practice with Speaker Notes
- `presentations/midterm_presentation/SPEAKER_NOTES.md` ko samne rakhkar PowerPoint slides ko full screen me run karein.
- Bol-bol kar 2 baar practice karein (sirf 8–10 minute ka presentation hoga).

### Day 13 & 14: Top 5 Tough Viva Questions
1. **"PINN me points kaise choose karte ho?"**
   - *Ans:* Domain ke andar random points (Latin Hypercube Sampling ya uniform random) lete hain jinko Collocation points kehte hain.
2. **"Agar loss 0 ho gaya toh iska kya matlab hai?"**
   - *Ans:* Iska matlab neural network ne PDE aur boundary conditions ko 100% accurately satisfy kar diya hai.
3. **"FDM se PINN kyu behtar laga?"**
   - *Ans:* FDM me grid points fix hote hain, jabki PINN continuous function seekhta hai. PINN ne 49% lower relative $L_2$ error diya.
4. **"Activation function kaunsa use kiya aur kyu?"**
   - *Ans:* $\tanh$ (Hyperbolic Tangent), kyunki yeh infinitely differentiable ($C^\infty$) hai, jo 2nd-order derivatives ke liye zaroori hai. ReLU use karne par 2nd derivative 0 ho jata!
5. **"Midterm ke baad agla step kya hoga?"**
   - *Ans:* Inverse problem—data se physical parameters (jaise viscosity $\nu$) discover karna.

### Day 15: Final Confidence Check
- Formal dress rehearse, slides ready on laptop/pen drive, printout of `reports/MIDTERM_PROGRESS_REPORT.md` (optional backup).
- Be calm and confident!
