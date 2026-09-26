"""
Script to generate a publication-quality Mathematical Architecture Diagram
for Physics-Informed Neural Networks (PINNs).
Saves to: reports/figures/pinn_mathematical_architecture.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_pinn_architecture():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Styling helpers
    box_style_nn = dict(boxstyle="round,pad=0.5", fc="#E3F2FD", ec="#1565C0", lw=2)
    box_style_ad = dict(boxstyle="round,pad=0.5", fc="#FFF3E0", ec="#E65100", lw=2)
    box_style_pde = dict(boxstyle="round,pad=0.5", fc="#E8F5E9", ec="#2E7D32", lw=2)
    box_style_loss = dict(boxstyle="round,pad=0.5", fc="#FCE4EC", ec="#C2185B", lw=2)
    box_style_opt = dict(boxstyle="round,pad=0.5", fc="#EDE7F6", ec="#4527A0", lw=2)
    
    # Title
    ax.text(7, 7.6, "Mathematical Architecture of Physics-Informed Neural Networks (PINN)",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#102C57")
    ax.text(7, 7.2, r"Continuous Functional Optimization via Automatic Differentiation ($\mathcal{J}(\theta) = \lambda_u \mathcal{L}_{data} + \lambda_f \mathcal{L}_{pde}$)",
            ha="center", va="center", fontsize=11, fontstyle="italic", color="#424242")
    
    # 1. Inputs
    ax.text(1.2, 5.0, "Spatiotemporal\nCoordinates\n\n" + r"$(x, t) \in Q_T$",
            ha="center", va="center", bbox=box_style_nn, fontsize=11, fontweight="bold")
    
    # 2. Neural Network
    ax.text(4.2, 5.0, "Deep Neural Network\nApproximator\n\n" + r"$\hat{u}(x, t; \theta) = \mathcal{W}_L \circ \sigma \dots \circ \mathcal{W}_1(x, t)$" + "\n\n" + r"$\sigma(z) = \tanh(z) \in C^\infty$",
            ha="center", va="center", bbox=box_style_nn, fontsize=10)
    
    # 3. Output Field
    ax.text(7.6, 5.0, "Predicted Solution\nField\n\n" + r"$\hat{u}(x, t)$",
            ha="center", va="center", bbox=box_style_nn, fontsize=11, fontweight="bold")
    
    # 4. PyTorch Autograd Engine
    ax.text(7.6, 2.2, "Automatic Differentiation (Autograd)\nChain Rule on Computational Graph\n\n" + r"$\frac{\partial \hat{u}}{\partial t}, \quad \frac{\partial \hat{u}}{\partial x}, \quad \frac{\partial^2 \hat{u}}{\partial x^2}$",
            ha="center", va="center", bbox=box_style_ad, fontsize=10, fontweight="bold")
    
    # 5. PDE Residual Operator
    ax.text(11.2, 2.2, "Nonlinear PDE Residual\n\n" + r"$f(x, t) = \hat{u}_t + \hat{u} \hat{u}_x - \nu \hat{u}_{xx}$" + "\n\n" + r"Target: $f(x, t) \to 0$",
            ha="center", va="center", bbox=box_style_pde, fontsize=10, fontweight="bold")
    
    # 6. Composite Loss Functional
    ax.text(11.2, 5.0, "Composite Multi-Objective Loss\n\n" + r"$\mathcal{L}_{total} = \mathcal{L}_{data} + \mathcal{L}_{pde}$" + "\n\n" + r"$\mathcal{L}_{pde} = \frac{1}{N_f}\sum |f(x_f, t_f)|^2$" + "\n" + r"$\mathcal{L}_{data} = \frac{1}{N_u}\sum |\hat{u} - u^*|^2$",
            ha="center", va="center", bbox=box_style_loss, fontsize=9.5)
    
    # 7. Optimizer & Backprop
    ax.text(4.2, 1.2, "Optimization Algorithm\nAdam $\\to$ L-BFGS\n" + r"$\theta^{(k+1)} = \theta^{(k)} - \eta_k H^{-1} \nabla_\theta \mathcal{L}$",
            ha="center", va="center", bbox=box_style_opt, fontsize=10, fontweight="bold")
    
    # Arrows
    arrow_props = dict(arrowstyle="->", lw=2, color="#1565C0")
    arrow_ad = dict(arrowstyle="->", lw=2, color="#E65100")
    arrow_pde = dict(arrowstyle="->", lw=2, color="#2E7D32")
    arrow_loss = dict(arrowstyle="->", lw=2, color="#C2185B")
    arrow_opt = dict(arrowstyle="->", lw=2, color="#4527A0", linestyle="--")
    
    # (x, t) -> NN
    ax.annotate("", xy=(2.6, 5.0), xytext=(2.0, 5.0), arrowprops=arrow_props)
    # NN -> u_hat
    ax.annotate("", xy=(6.8, 5.0), xytext=(5.8, 5.0), arrowprops=arrow_props)
    
    # u_hat -> Autograd
    ax.annotate("", xy=(7.6, 3.2), xytext=(7.6, 4.2), arrowprops=arrow_ad)
    # Autograd -> Residual
    ax.annotate("", xy=(9.7, 2.2), xytext=(9.3, 2.2), arrowprops=arrow_pde)
    # Residual -> Loss
    ax.annotate("", xy=(11.2, 3.8), xytext=(11.2, 3.2), arrowprops=arrow_loss)
    # u_hat -> Loss (Boundary/Data)
    ax.annotate("", xy=(9.8, 5.0), xytext=(8.4, 5.0), arrowprops=arrow_loss)
    
    # Loss -> Optimizer (Feedback loop)
    ax.annotate("", xy=(6.0, 1.2), xytext=(10.0, 4.0), arrowprops=arrow_opt)
    # Optimizer -> NN (Weight update)
    ax.annotate("", xy=(4.2, 3.8), xytext=(4.2, 2.0), arrowprops=arrow_opt)
    ax.text(3.3, 2.9, r"Update $\theta=\{W, b\}$", color="#4527A0", fontsize=9, fontweight="bold")
    
    plt.tight_layout()
    os.makedirs("reports/figures", exist_ok=True)
    save_path = "reports/figures/pinn_mathematical_architecture.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Architecture diagram saved to: {save_path}")

if __name__ == "__main__":
    draw_pinn_architecture()
