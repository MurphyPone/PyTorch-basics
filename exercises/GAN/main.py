import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np 
import pandas as pd 
import seaborn as sns 
from IPython.display import clear_output
sns.set()
sns.set(rc={'figure.figsize':(11.7, 8.27)})
torch.__version__

# Data parameters 
data_mean = 4
data_stddev = 1.25

# Model parameters
g_input_sz      = 1     # Random noise dimension coming into gnerator, per output vector
g_hidden_sz     = 50    # Generator complexity
g_output_sz     = 1     # Size of generated output vector

d_input_sz      = 100   # Minibatch size - cardinality of distributions TODO ask what this means 
d_hidden_sz     = 50    # Discriminator complexity 
d_output_sz     = 1     # Single dimension for 'real' vs. 'fake'

minibatch_sz    = d_input_sz

g_lr = 1e-4
d_lr = 2e-4
num_epochs = 10000
print_interval = 200
d_steps = 1 # N.B. These frequencies do not have to be consistent, d can train faster than g  
g_steps = 1 

def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1,n)))   # Make a tensor holding mu, sigma, in (1,n) dimensions

def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n) #  [m * [n elements],]  

class Generator(nn.Module):
    def __init__(self, input_sz, hidden_sz, output_sz):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_sz, hidden_sz)
        self.map2 = nn.Linear(hidden_sz, hidden_sz)
        self.map3 = nn.Linear(hidden_sz, output_sz)

    def forward(self, x):
        x = F.elu(self.map1(x))           # Element-wise Linearization Unit
        x = F.elu(self.map2(x))
        x = self.map3(x)
        return x 

class Discriminator(nn.Module):
    def __init__(self, input_sz, hidden_sz, output_sz):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_sz, hidden_sz)
        self.map2 = nn.Linear(hidden_sz, hidden_sz)
        self.map3 = nn.Linear(hidden_sz, output_sz)

    def forward(self, x):
        x = F.elu(self.map1(x))           # Element-wise Linearization Unit
        x = F.elu(self.map2(x))
        x = self.map3(x)
        x = torch.sigmoid(x).squeeze(0)     # Why does the the Discriminator constrain to a sigmoid?
        return x 

G = Generator(input_sz=g_input_sz, hidden_sz=g_hidden_sz, output_sz=g_output_sz)
D = Discriminator(input_sz=d_input_sz, hidden_sz=d_hidden_sz, output_sz=d_output_sz)

def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]

d_sampler = get_distribution_sampler(data_mean, data_stddev)
gi_sampler = get_generator_input_sampler()

criterion = nn.BCELoss()    # Binary cross entropy

d_optimizer = optim.Adam(D.parameters(), lr=d_lr)
g_optimizer = optim.Adam(G.parameters(), lr=g_lr)

observations = []

for epoch in range(num_epochs):
    for d_index in range(d_steps):
        # Step 1: Tran D on real+fake data
        D.zero_grad()   # inherited from nn.Module?

        # Step 1.a: Train D on real 
        d_real_data = d_sampler(d_input_sz)
        d_real_decision = D(d_real_data)
        d_real_error = criterion(d_real_decision, torch.ones(1)) # torch.ones(1) --> [[1]]
        d_real_error.backward()

        # Step 1.b: Train D on fake 
        d_gen_input = gi_sampler(minibatch_sz, g_input_sz)
        d_fake_data = G(d_gen_input).detach()   # "Detach to avoid training G on these labels" ??
        d_fake_decision = D(d_fake_data.t())    # .t() accepts a 2D tensor and transposes dimensions 0 and 1
        d_fake_error = criterion(d_fake_decision, torch.zeros(1)) # torch.ones(1) --> [[0]] TODO why use zeros here?
        d_fake_error.backward()

        d_optimizer.step()

    for g_index in range(g_steps):
        # Step 2. Train G on D's response (but DO NOt train D on these labels)
        G.zero_grad()

        gen_input = gi_sampler(minibatch_sz, g_input_sz)
        g_fake_data = G(gen_input)
        dg_fake_decision = D(g_fake_data.t())
        g_error = criterion(dg_fake_decision, torch.ones(1)) # trying to fool D 

        g_error.backward()
        g_optimizer.step()

    if epoch % print_interval == 0:
        clear_output(wait=True)
        print("epoch %d/%d" % (epoch, num_epochs))
        observations.append((epoch,
                            extract(d_real_error)[0],
                            extract(d_fake_error)[0],
                            extract(g_error)[0],
                            stats(extract(d_real_data))[0],
                            stats(extract(d_real_data))[1],
                            stats(extract(d_fake_data))[0],
                            stats(extract(d_fake_data))[1]))

df = pd.DataFrame(observations, columns=['epoch',
                                         'real_error',
                                         'fake_error',
                                         'g_error',
                                         'real_mean',
                                         'real_std',
                                         'fake_mean',
                                         'fake_std'])
melted = df.melt('epoch', var_name='cols', value_name='vals') # Unpivot a DataFrame from wide format to long format, optionally leaving identifier variables set.

# plot and save figure 
sns_ln_plt = sns.lineplot(x='epoch', y='vals', hue='cols', data=melted)
fig = sns_ln_plt.get_figure()
fig.savefig("lineplot.png")


fake_samples = []
real_samples = []

with torch.no_grad():
    for i in range(100):
        d_real_data = extract(d_sampler(d_input_sz))
        d_gen_input = gi_sampler(minibatch_sz, g_input_sz)
        d_fake_data = extract(G(d_gen_input))

        fake_samples = fake_samples + d_fake_data
        real_samples = real_samples + d_real_data

fig.clf()
sns.distplot(real_samples)
sns_dist_plt = sns.distplot(fake_samples)
fig = sns_dist_plt.get_figure()
fig.savefig("distplot.png")

print("fake_mean: ", np.array(fake_samples).mean())
print("real_mean: ", np.array(real_samples).mean())
print("fake_std: ", np.array(fake_samples).std())
print("fake_std: ", np.array(real_samples).std())