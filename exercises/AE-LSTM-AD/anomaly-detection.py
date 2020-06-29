from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import time 
import copy
import shutil
import torch
from torch import nn, optim 
import torch.nn.functional as F 
from torchvision import datasets, transforms, models 
from arff2pandas import a2p
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set(style='whitegrid', palette='muted', font_scale=1.2)
# PALLETTE = ["#01BEFE", "#FFD00", "#FF7D00", "#FF006D", "8F00FF"]
# sns.set_palette(sns.color_palette(PALLETTE))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using {device}")

RANDOM_SEED = 86
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Load the data
with open("data/ECG5000_TRAIN.arff") as f:
    train = a2p.load(f)

with open("data/ECG5000_TEST.arff") as f:
    test = a2p.load(f)
# print(train.head(), test.head())

# combining the dataset since we're not doing classification and just want maxmimal data
df = train.append(test) 
df = df.sample(frac=1.0) # shuffle the data for ... TODO ?

# 5 classes:
#   1 Normal (N)    (everything else is anomalous) <-- this is the training set
#   2 R-on-T Premature Ventricular Contraction (R-on-T PVC)
#   3 Premature Ventricular Contraction (PVC)
#   4 Supra-ventricular PRemature Ectopic Beat (SP or EB)
#   5 Unclassified Beat (UB)

# Data preprocessing
CLASS_NORMAL = 1
class_names = ['N', 'R-on-T', 'PVC', 'SP', 'UB']
new_cols = list(df.columns)
new_cols[-1] = 'target'
df.columns = new_cols

# Exploration 
#print(df.target.value_counts())
    # 1    2919
    # 2    1767
    # 4     194
    # 3      96
    # 5      24

# ax = sns.countplot(df.target)
# ax.set_xticklabels(class_names)

# util class to plot the classes' means, and std_devs
def plot_time_series_class(data, class_name, ax, n_steps=10): 
    time_series_df = pd.DataFrame(data)
    smooth_path = time_series_df.rolling(n_steps).mean()
    path_deviation = 2 * time_series_df.rolling(n_steps).std()

    under_line = (smooth_path - path_deviation)[0]
    over_line = (smooth_path + path_deviation)[0]

    ax.plot(smooth_path, linewidth=2)
    ax.fill_between(
        path_deviation.index,
        under_line,
        over_line,
        alpha=.15
    )
    ax.set_title(class_name)

classes = df.target.unique()
fig, axs = plt.subplots(
    nrows=len(classes) // 3 + 1,
    ncols=3,
    sharey=True,
    figsize=(10, 6) 
)

for i, cls in enumerate(classes):
    ax = axs.flat[i]
    data = df[df.target == cls].drop(labels='target', axis=1).mean(axis=0).to_numpy()
    plot_time_series_class(data, class_names[i], ax)

fig.delaxes(axs.flat[-1])
fig.tight_layout()
# plt.show()
# plt.savefig("df")

# More data preprocessing
normal_df = df[df.target == str(CLASS_NORMAL)].drop(labels='target', axis=1)
anomalous_df = df[df.target != str(CLASS_NORMAL)].drop(labels='target', axis=1)

# partitionining data into train, validation, and testing subsets
train_df, val_df = train_test_split(normal_df, test_size=0.15, random_state=RANDOM_SEED)
val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=RANDOM_SEED)

train_seq = train_df.astype(np.float32).to_numpy().tolist()
test_seq = test_df.astype(np.float32).to_numpy().tolist()
val_seq = val_df.astype(np.float32).to_numpy().tolist()
anomalous_seq = anomalous_df.astype(np.float32).to_numpy().tolist()

# converts to torch tensors
def create_dataset(sequences):
    dataset = [torch.tensor(seq).unsqueeze(1) for seq in sequences]
    
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features # don't care about n_seq

train_dataset, seq_len, n_features  = create_dataset(train_seq)
val_dataset, _, _                   = create_dataset(val_seq)
test_normal_dataset, _, _          = create_dataset(test_seq)
test_anomolous_dataset, _, _        = create_dataset(anomalous_seq)

# LSTM Auto-Encoder
class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)

        return hidden_n.reshape((1, self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, output_dim=1):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = 2 * input_dim
        self.output = input_dim

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.dense_layers = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
       x = x.repeat(self.seq_len, 1)
       x = x.reshape((1, self.seq_len, self.input_dim))
       x, (hidden_n, cell_n) = self.rnn1(x)
       x, (hidden_n, cell_n) = self.rnn2(x)
       x = x.reshape((self.seq_len, self.hidden_dim))
       
       return self.dense_layers(x)

# Recurrent AE
class RAE(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RAE, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = RAE(seq_len, n_features, embedding_dim=128)
model = model.to(device)


# Training 
def train_model(model, train_data, val_data, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(device)

    history = dict(train=[], val=[])
    for epoch in range(1, n_epochs+1):
        model = model.train() 
        train_losses = [] 

        for seq_true in train_data:
            optimizer.zero_grad()
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)
            loss.backward() 
            optimizer.step()

            train_losses.append(loss.item())
    
    val_losses = []
    model = model.eval() 
    with torch.no_grad():
        for seq_true in val_data:
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)

            val_losses.append(loss.item())
    
    mean_train_losses = np.mean(train_losses)
    mean_val_loss = np.mean(val_losses)
    history['train'].append(mean_train_loss)
    history['val'].append(mean_val_loss)

    print(f'Epoch {epoch}: train loss {mean_train_losses} val loss {mean_val_loss}')

    return model.eval(), history

model, history = train_model(model, train_dataset, val_dataset, n_epochs=150)