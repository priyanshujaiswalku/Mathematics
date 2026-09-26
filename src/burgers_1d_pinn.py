"""
1D Viscous Burgers' Equation Forward Solver using Physics-Informed Neural Networks (PINN)
Problem Formulation (Raissi et al., 2017 Part I Benchmark):
    u_t + u * u_x - (0.01 / pi) * u_xx = 0,  x in [-1, 1], t in [0, 1]
    Initial Condition:  u(x, 0) = -sin(pi * x)
    Boundary Condition: u(-1, t) = u(1, t) = 0

Author: Priyanshu Kumar (M.Sc. Mathematics Dissertation)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Set reproducibility seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nu = 0.01 / np.pi

# -------------------------------------------------------------
# 1. PINN Architecture for Spatiotemporal PDE: (x, t) -> u
# -------------------------------------------------------------
class BurgersPINN(nn.Module):
    def __init__(self, layers=[2, 40, 40, 40, 1]):
        super(BurgersPINN, self).__init__()
        self.network = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.network.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = nn.Tanh()
        
    def forward(self, xt):
        out = xt
        for i in range(len(self.network) - 1):
            out = self.activation(self.network[i](out))
        out = self.network[-1](out)
        return out

def compute_burgers_residual(model, xt_colloc):
    """
    Computes PDE residual f(x, t) = u_t + u * u_x - nu * u_xx via Autograd
    """
    xt_colloc.requires_grad_(True)
    u = model(xt_colloc)
    
    # Gradient of u w.r.t (x, t)
    grads = torch.autograd.grad(
        u, xt_colloc,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]
    
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    
    # Second derivative: u_xx
    u_xx = torch.autograd.grad(
        u_x, xt_colloc,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True
    )[0][:, 0:1]
    
    # Residual
    f = u_t + u * u_x - nu * u_xx
    return f

# -------------------------------------------------------------
# 2. Data Generation & Collocation Sampling
# -------------------------------------------------------------
def generate_training_data(N_u=100, N_f=2500):
    # Initial condition: t = 0, x in [-1, 1]
    x_ic = np.random.uniform(-1, 1, (N_u // 2, 1))
    t_ic = np.zeros((N_u // 2, 1))
    u_ic = -np.sin(np.pi * x_ic)
    
    # Boundary conditions: x = -1 or +1, t in [0, 1]
    t_bc = np.random.uniform(0, 1, (N_u // 2, 1))
    x_bc = np.random.choice([-1.0, 1.0], size=(N_u // 2, 1))
    u_bc = np.zeros((N_u // 2, 1))
    
    # Combine IC and BC
    xt_u = np.vstack([np.hstack([x_ic, t_ic]), np.hstack([x_bc, t_bc])])
    u_data = np.vstack([u_ic, u_bc])
    
    # Collocation points inside domain [-1, 1] x [0, 1]
    x_f = np.random.uniform(-1, 1, (N_f, 1))
    t_f = np.random.uniform(0, 1, (N_f, 1))
    xt_f = np.hstack([x_f, t_f])
    
    return (
        torch.tensor(xt_u, dtype=torch.float32, device=device),
        torch.tensor(u_data, dtype=torch.float32, device=device),
        torch.tensor(xt_f, dtype=torch.float32, device=device)
    )

# -------------------------------------------------------------
# 3. Training Loop
# -------------------------------------------------------------
def train_burgers(epochs=1800, lr=1e-3):
    model = BurgersPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    xt_u, u_train, xt_f = generate_training_data(N_u=200, N_f=2000)
    loss_history = []
    start_time = time.time()
    
    print("\n--- Training 1D Burgers' Equation PINN ---")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Loss on IC and BC
        u_pred = model(xt_u)
        loss_u = torch.mean((u_pred - u_train) ** 2)
        
        # Loss on PDE physics residual
        f_pred = compute_burgers_residual(model, xt_f)
        loss_f = torch.mean(f_pred ** 2)
        
        loss = 2.0 * loss_u + loss_f
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 300 == 0 or epoch == epochs:
            print(f"Epoch {epoch:4d}/{epochs} | Total Loss: {loss.item():.6e} | Data Loss: {loss_u.item():.6e} | PDE Loss: {loss_f.item():.6e}")
            
    train_time = time.time() - start_time
    print(f"[INFO] Burgers PINN training completed in {train_time:.2f} seconds.")
    return model, loss_history

# -------------------------------------------------------------
# 4. Visualization & Spatiotemporal Plots
# -------------------------------------------------------------
def main():
    model, loss_history = train_burgers(epochs=1800, lr=1e-3)
    
    # Create evaluation meshgrid
    x = np.linspace(-1, 1, 200)
    t = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x, t)
    
    xt_eval = np.hstack([X.flatten()[:, None], T.flatten()[:, None]])
    xt_tensor = torch.tensor(xt_eval, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        u_pred = model(xt_tensor).cpu().numpy().reshape(X.shape)
        
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(16, 5))
    
    # 1. 2D Spatiotemporal Heatmap
    ax1 = fig.add_subplot(1, 2, 1)
    cp = ax1.contourf(T, X, u_pred, levels=80, cmap='rainbow')
    cbar = fig.colorbar(cp, ax=ax1)
    cbar.set_label('Predicted $u(x, t)$', fontsize=11)
    ax1.set_title("1D Burgers' Equation: Spatiotemporal Solution Field $u(x, t)$", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Time $t$", fontsize=11)
    ax1.set_ylabel("Space $x$", fontsize=11)
    
    # 2. Time Slice Snapshots (Shock wave steepening)
    ax2 = fig.add_subplot(1, 2, 2)
    t_slices = [0.0, 0.25, 0.50, 0.75, 1.0]
    colors = ['navy', 'darkgreen', 'orange', 'red', 'purple']
    
    for t_val, col in zip(t_slices, colors):
        t_idx = int(t_val * (len(t) - 1))
        ax2.plot(x, u_pred[t_idx, :], label=f'$t = {t_val:.2f}$', color=col, linewidth=2.0)
        
    ax2.set_title("Temporal Evolution & Steep Gradient Formation (Shock Wave)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Space $x$", fontsize=11)
    ax2.set_ylabel("Velocity $u(x, t)$", fontsize=11)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs("reports/figures", exist_ok=True)
    save_path = "reports/figures/burgers_pinn_solution.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] High-resolution Burgers figure saved to: {save_path}")

if __name__ == "__main__":
    main()
