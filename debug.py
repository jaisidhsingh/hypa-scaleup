import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducibility
torch.manual_seed(42)

# Create a 200x200 matrix
weight = nn.Parameter(torch.randn(200, 200))
fixed_proj = torch.randn(200, 200)
full_matrix = fixed_proj @ weight

# Create a 100-sized vector
vector = torch.randn(100)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD([weight], lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass using only the submatrix
    output = torch.mv(full_matrix[:100, :100], vector)

    # Compute loss
    target = torch.randn(100)  # Random target for demonstration
    loss = criterion(output, target)

    # Backward pass
    loss.backward()

    # Zero out gradients for unused parts
    with torch.no_grad():
        full_matrix.grad[100:, :] = 0
        full_matrix.grad[:, 100:] = 0

    # Optimizer step
    optimizer.step()

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Verify final loss
with torch.no_grad():
    final_output = torch.mv(full_matrix[:100, :100], vector)
    final_loss = criterion(final_output, target)
    print(f'Final Loss: {final_loss.item():.4f}')

# Check gradients
print("Gradient shape:", full_matrix.grad.shape)
print("Sum of gradients in used part:", full_matrix.grad[:100, :100].sum().item())
print("Sum of gradients in unused part:", full_matrix.grad[100:, 100:].sum().item())
