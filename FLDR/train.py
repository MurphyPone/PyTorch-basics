import torch 

from distribution import Distribution
from visualize import * 

n_epochs = 10000
lr = 0.05
batch_size = 64
vis_iter = 200

target = Distribution([.1, .3, .4, .1, .1], [-3, 3, 15, 10 , -10], [1, 2, 3, 1, 5])
Q = Distribution(K=10, requires_grad=True)

plot_dist(target, 'target', '#ff8200', (-20, 20))
plot_dist(Q, 'Q', '#7600fe', (-20, 20))

optimizer = torch.optim.Adam(Q.parameters(), lr=lr)

for epoch in range(int(n_epochs)):
    x = target.sample(batch_size)
    loss = torch.pow(Q(x) - target(x), 2).mean() # MSE

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % vis_iter == vis_iter - 1:
        plot_loss(epoch, loss)
        plot_dist(Q, 'Q', '#7600fe', (-20, 20))
