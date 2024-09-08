import torch
import torch.nn as nn

d = 5
d_out = 2
x = torch.randn(1, d)
y = torch.randn(1, d_out)
c = 3
hypernet = nn.Linear(c, d_out*d*2)  # predict two times more as example
optimizer = torch.optim.Adam(hypernet.parameters(), lr=1e-3)
f = nn.Linear(d, d_out)

criterion = nn.MSELoss()

for i in range(2):
    optimizer.zero_grad()

    w_pred = hypernet(torch.randn(1, c)).reshape(d_out, d*2)
    w_pred = w_pred[:, :d]  # slice

    key = 'weight'
    f.__dict__[key] = w_pred  # set the value avoiding the internal logic of PyTorch
    f._parameters[key] = w_pred
    y_ = x @ w_pred.T

    loss = criterion(y_, y)

    loss.backward()

    optimizer.step()

    loss.item()
    print(hypernet.weight.grad.norm(), f.weight.grad)
