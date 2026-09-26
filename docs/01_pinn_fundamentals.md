# Physics-Informed Neural Networks (PINNs): Fundamentals & Intuition

**Author:** Priyanshu Kumar  
**Target Audience:** Mathematics Dissertation Research  

---

## 💡 1. PINN Kya Hai? (In Simple Words)

Traditional Neural Networks sirf **Data** dekh kar seekhte hain (jaise cat vs dog images, ya stock prices). Agar unke paas data kam ho ya noisy ho, toh wo physical reality se bhatak jaate hain (unrealistic predictions dene lagte hain).

**PINN (Physics-Informed Neural Network)** me hum neural network ko sirf data nahi dete, balki **Physics ke niyam (Partial Differential Equations)** bhi sikhate hain:

$$\text{Loss Total} = \underbrace{\text{Loss}_{\text{data}}}_{\text{Data par error}} + \underbrace{\text{Loss}_{\text{physics}}}_{\text{PDE ke niyam todne par penalty}}$$

> **Simple Analogy:**  
> Socho ek student maths ka sawal solve kar raha hai. 
> - $\text{Loss}_{\text{data}}$ yeh check karta hai ki kya usne question ke boundary points (given values) sahi likhe hain?
> - $\text{Loss}_{\text{physics}}$ yeh check karta hai ki kya steps ke dauran usne formula (PDE) sahi follow kiya hai ya nahi?

---

## ⚙️ 2. PINN Kaise Kaam Karta Hai? (Mathematical Architecture)

Ek standard PDE problem ko dekhte hain (Jaise 1D Burgers' Equation ya Heat Equation):

$$\mathcal{N}[u](x, t) = 0 \quad \text{for } x \in \Omega, \ t \in [0, T]$$

Jahan:
- $x$ space coordinate hai, $t$ time coordinate hai.
- $u(x, t)$ unknown solution hai jisko hume dhoondhna hai.

### Step 1: Neural Network as a Function Approximator
Hum ek Feed-Forward Neural Network (MLP) banate hain jiska:
- **Input:** Coordinates $(x, t)$
- **Output:** Predicted value $\hat{u}(x, t; \theta)$, jahan $\theta = \{W, b\}$ weights aur biases hain.

$$\hat{u} = \text{NN}(x, t; \theta)$$

---

### Step 2: Automatic Differentiation (The Secret Weapon)
Classical Numerical Methods (FDM) me derivative nikalne ke liye grid banani padti hai:
$$\frac{\partial u}{\partial x} \approx \frac{u(x+\Delta x) - u(x)}{\Delta x} \quad (\text{Approximation with truncation error})$$

Lekin PINNs me hum **Automatic Differentiation (AD / PyTorch Autograd)** use karte hain. AD chain rule of calculus use karke **Exact Derivative** nikalta hai:

$$\frac{\partial \hat{u}}{\partial x} = \text{Exact analytical derivative computed via computational graph}$$

Iska matlab: **Hume koi grid (mesh) banane ki zaroorat nahi padti!** (Mesh-free method).

---

### Step 3: PDE Residual Definition
Agar hamara PDE hai:
$$f(x, t) := \frac{\partial \hat{u}}{\partial t} + \hat{u} \frac{\partial \hat{u}}{\partial x} - \nu \frac{\partial^2 \hat{u}}{\partial x^2}$$

Agar neural network ka prediction $\hat{u}$ 100% exact physics follow karega, toh $f(x, t)$ ki value **0** honi chahiye.
Agar $f(x, t) \neq 0$, toh iska matlab physics follow nahi ho rahi hai!

---

### Step 4: Composite Loss Function (Optimization Objective)

Hum network ko train karne ke liye 2 tarah ke points sample karte hain:
1. **Initial & Boundary Points ($N_u$ points):** Jahan hume solution pehle se pata hai.
   $$MSE_u = \frac{1}{N_u} \sum_{i=1}^{N_u} \left| \hat{u}(x_u^i, t_u^i) - u^i \right|^2$$

2. **Collocation Points ($N_f$ points):** Domain ke andar random points jahan sirf PDE residual 0 hona chahiye.
   $$MSE_f = \frac{1}{N_f} \sum_{i=1}^{N_f} \left| f(x_f^i, t_f^i) \right|^2$$

Total Loss:
$$\mathcal{L}(\theta) = MSE_u + MSE_f$$

Backpropagation aur optimizers (Adam + L-BFGS) use karke weights $\theta$ ko update kiya jaata hai jab tak loss minimize na ho jaye.

---

## 🔍 3. Forward vs Inverse Problems (Part I vs Part II)

| Feature | Forward Problem (Part I Paper) | Inverse Problem (Part II Paper) |
| :--- | :--- | :--- |
| **Goal** | Given PDE & coefficients, find solution $u(x, t)$ | Given sparse observations of $u$, find unknown coefficients (e.g. viscosity $\nu$) |
| **Analogy** | Formula pata hai, answer nikalna hai | Experiment ka answer pata hai, secret formula/parameter dhoondhna hai |
| **Unknowns** | Only network weights $\theta$ | Network weights $\theta$ **PLUS** physical parameters (e.g. $\nu, \lambda_1, \lambda_2$) |

---

## 🎯 4. Supervisor First Meeting: Expected Questions & Confident Answers

### Q1: *"Tum neural network kyu use kar rahe ho jab hamare paas classical FDM/FEM methods pehle se hain?"*
> **Answer:**  
> *"Sir, classical methods (FDM/FEM) mesh par depend karte hain. High dimensions me ya irregular geometries me mesh banana computationally bahut expensive hota hai (Curse of Dimensionality).  
> PINNs **mesh-free** hain aur continuous space-time par direct solution dete hain. Saath hi, classical methods inverse problems (parameter estimation from noisy data) ke liye bahut complex hote hain, jabki PINNs me inverse problem natural way me solve ho jata hai."*

### Q2: *"Physics neural network ke andar kaise enter karti hai?"*
> **Answer:**  
> *"Sir, physics loss function ke zariye enter karti hai. Hum PyTorch ke Automatic Differentiation se PDE residual formulate karte hain, aur us residual ko penalize karte hain ($MSE_f$). Is tarah network unhi weights ko select karta hai jo differential equation ko satisfy karein."*

### Q3: *"Aapka agle 10 dino ka kya plan hai?"*
> **Answer:**  
> *"Sir, maine literature review aur mathematical formulation ready kar liya hai. Agle 10 dino me main PyTorch me 1D Poisson/Heat equation ka working model implement karke exact analytical solution ke saath error comparison dikhaunga."*
