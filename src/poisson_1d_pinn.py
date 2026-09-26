"""
1D Poisson Equation Benchmark: PINN vs Classical Finite Difference Method (FDM)
Problem Formulation:
    -d^2u/dx^2 = pi^2 * sin(pi * x),  x in [-1, 1]
    Boundary Conditions: u(-1) = 0, u(1) = 0
    Exact Analytical Solution: u*(x) = sin(pi * x)

Author: Priyanshu Kumar (M.Sc. Mathematics Dissertation)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

# -------------------------------------------------------------
# 1. Classical Numerical Solver: Finite Difference Method (FDM)
# -------------------------------------------------------------
def solve_fdm(N=50):
    """
    Solves -u''(x) = pi^2 * sin(pi * x) on [-1, 1] using 2nd order Central Difference.
    """
    x = np.linspace(-1, 1, N)
    dx = x[1] - x[0]
    
    # Interior points
    x_int = x[1:-1]
    n_int = len(x_int)
    
    # Tridiagonal matrix A for -d^2/dx^2: [-1, 2, -1] / dx^2
    main_diag = 2.0 * np.ones(n_int) / (dx ** 2)
    off_diag = -1.0 * np.ones(n_int - 1) / (dx ** 2)
    A = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
    
    # Source term rhs = pi^2 * sin(pi * x)
    b = (np.pi ** 2) * np.sin(np.pi * x_int)
    
    # Solve linear system A u = b
    u_int = np.linalg.solve(A, b)
    
    # Full solution including BCs u(-1) = 0, u(1) = 0
    u_fdm = np.zeros(N)
    u_fdm[1:-1] = u_int
    
    u_exact = np.sin(np.pi * x)
    rel_l2 = np.linalg.norm(u_fdm - u_exact) / np.linalg.norm(u_exact)
    return x, u_fdm, rel_l2

# -------------------------------------------------------------
# 2. Physics-Informed Neural Network (PINN) Model
# -------------------------------------------------------------
class PoissonPINN(nn.Module):
    def __init__(self, layers=[1, 32, 32, 1]):
        super(PoissonPINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = nn.Tanh()
        
    def forward(self, x):
        out = x
        for i in range(len(self.layers) - 1):
            out = self.activation(self.layers[i](out))
        out = self.layers[-1](out)
        return out

def compute_pde_residual(model, x_colloc):
    """
    Computes residual f(x) = d^2u/dx^2 + pi^2 * sin(pi * x) using torch.autograd
    """
    x_colloc.requires_grad_(True)
    u = model(x_colloc)
    
    # First derivative: du/dx
    du_dx = torch.autograd.grad(
        u, x_colloc,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Second derivative: d^2u/dx^2
    d2u_dx2 = torch.autograd.grad(
        du_dx, x_colloc,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # PDE Residual: -u''(x) - pi^2 * sin(pi * x) = 0  =>  u''(x) + pi^2 * sin(pi * x) = 0
    f = d2u_dx2 + (np.pi ** 2) * torch.sin(np.pi * x_colloc)
    return f

# -------------------------------------------------------------
# 3. Training Routine
# -------------------------------------------------------------
def train_pinn(epochs=1200, lr=1e-3, N_f=100):
    model = PoissonPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Boundary points: x = -1, x = 1, u = 0
    x_bc = torch.tensor([[-1.0], [1.0]], dtype=torch.float32, device=device)
    u_bc = torch.tensor([[0.0], [0.0]], dtype=torch.float32, device=device)
    
    # Collocation points: uniformly sampled inside (-1, 1)
    x_f = torch.linspace(-1, 1, N_f, dtype=torch.float32, device=device).unsqueeze(1)
    
    loss_history = []
    start_time = time.time()
    
    print("\n--- Training PINN via PyTorch Autograd ---")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Boundary loss
        u_bc_pred = model(x_bc)
        loss_bc = torch.mean((u_bc_pred - u_bc) ** 2)
        
        # PDE residual loss
        f_pred = compute_pde_residual(model, x_f)
        loss_pde = torch.mean(f_pred ** 2)
        
        # Total loss
        loss = 10.0 * loss_bc + loss_pde  # Weight BC slightly higher for rapid enforcement
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 200 == 0 or epoch == epochs:
            print(f"Epoch {epoch:4d}/{epochs} | Total Loss: {loss.item():.6e} | PDE Loss: {loss_pde.item():.6e} | BC Loss: {loss_bc.item():.6e}")
            
    train_time = time.time() - start_time
    print(f"[INFO] PINN Training completed in {train_time:.2f} seconds.")
    return model, loss_history, train_time

# -------------------------------------------------------------
# 4. Evaluation and Visualization
# -------------------------------------------------------------
def main():
    # Solve with FDM
    x_fdm, u_fdm, rel_l2_fdm = solve_fdm(N=50)
    
    # Train PINN
    model, loss_history, pinn_time = train_pinn(epochs=1200, lr=1e-3, N_f=100)
    
    # Test PINN on high-resolution grid (N=200)
    x_test_np = np.linspace(-1, 1, 200)
    x_test_tensor = torch.tensor(x_test_np, dtype=torch.float32, device=device).unsqueeze(1)
    with torch.no_grad():
        u_pinn_pred = model(x_test_tensor).cpu().numpy().flatten()
    
    u_exact = np.sin(np.pi * x_test_np)
    rel_l2_pinn = np.linalg.norm(u_pinn_pred - u_exact) / np.linalg.norm(u_exact)
    abs_error_pinn = np.abs(u_pinn_pred - u_exact)
    
    print("\n=======================================================")
    print("                BENCHMARK COMPARISON RESULTS           ")
    print("=======================================================")
    print(f"Finite Difference Method (FDM, N=50) Relative L2 Error : {rel_l2_fdm:.4e}")
    print(f"Physics-Informed Neural Network (PINN) Relative L2 Error: {rel_l2_pinn:.4e}")
    print(f"Max Absolute Error (PINN)                              : {np.max(abs_error_pinn):.4e}")
    print("=======================================================\n")
    
    # Plotting Publication-Quality Figures
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot 1: Solution Comparison
    axs[0].plot(x_test_np, u_exact, 'k-', linewidth=2.5, label=r'Exact: $\sin(\pi x)$')
    axs[0].plot(x_test_np, u_pinn_pred, 'r--', linewidth=2.0, label='PINN Prediction')
    axs[0].scatter(x_fdm[::3], u_fdm[::3], color='blue', s=30, alpha=0.7, label='FDM (N=50 sampled)')
    axs[0].set_title("Solution Profiles: Exact vs PINN vs FDM", fontsize=13, fontweight='bold')
    axs[0].set_xlabel("x", fontsize=12)
    axs[0].set_ylabel("u(x)", fontsize=12)
    axs[0].legend(fontsize=10, loc='best')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    
    # Subplot 2: Pointwise Absolute Error
    axs[1].semilogy(x_test_np, abs_error_pinn, 'r-', linewidth=2.0, label=r'$|u_{\mathrm{PINN}} - u^*|$')
    axs[1].set_title("Pointwise Error Distribution (PINN)", fontsize=13, fontweight='bold')
    axs[1].set_xlabel("x", fontsize=12)
    axs[1].set_ylabel(r"Absolute Error $|u - u^*|$", fontsize=12)
    axs[1].legend(fontsize=10)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    
    # Subplot 3: Training Loss Convergence
    axs[2].semilogy(loss_history, 'b-', linewidth=1.5, label='Composite Loss')
    axs[2].set_title("PINN Training Loss Convergence", fontsize=13, fontweight='bold')
    axs[2].set_xlabel("Epoch", fontsize=12)
    axs[2].set_ylabel(r"Total Loss $\mathcal{L}(\theta)$", fontsize=12)
    axs[2].legend(fontsize=10)
    axs[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs("reports/figures", exist_ok=True)
    save_path = "reports/figures/poisson_pinn_vs_fdm.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] High-resolution benchmark figure saved to: {save_path}")

if __name__ == "__main__":
    main()
