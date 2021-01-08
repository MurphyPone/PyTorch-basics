import torch 
import torch.nn as nn
import torch.optim as optim 
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np 
import random 
import spacy 

from model import * 
from visualize import *

algo_name = 'Translation Transformer'
epoch_colors = ['#f00', '#f90', '#080', '#088', '#00f', '#91f'] 
smooth_plots = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def translate_sentence(model, sentence, german, english, device, max_len=100):
  spacy_de = spacy.load("de") # load DE tokenizer
  
  if type(sentence) == str:
    tokens = [token.text.lower() for token in spacy_de(sentence)]
  else:
    tokens = [token.lower() for token in sentence]

  # bookend with SOS, EOS
  tokens.insert(0, german.init_token)
  tokens.append(german.eos_token)

  # convert tokens to indices
  text_to_indices = [german.vocab.stoi[token] for token in tokens]

  # convert to tensor
  sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

  outputs = [english.vocab.stoi["<sos>"]]
  for i in range(max_len):
    trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

    with torch.no_grad():
      output = model(sentence_tensor, trg_tensor)

    best_guess = output.argmax(2)[-1, :].item()
    outputs.append(best_guess)
    
    if best_guess == english.vocab.stoi["<eos>"]:
      break

  translated_sentence = [english.vocab.itos[idx] for idx in outputs]
  return translated_sentence


# install spacy lang: python -m spacy download [en/de]

spacy_de = spacy.load("de")
spacy_en = spacy.load("en")

def tokenize_de(text):
  return [token.text for token in spacy_de.tokenizer(text)]

def tokenize_en(text):
  return [token.text for token in spacy_en.tokenizer(text)]

german  = Field(tokenize=tokenize_de, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_en, lower=True, init_token="<sos>", eos_token="<eos>")

train_data, validation_data, test_data = Multi30k.splits(
  exts=(".de", ".en"), fields=(german, english)
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


# hyper params
n_epochs   = 5
lr         = 3e-4
batch_size = 32 

# model params
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embedding_size = 512
num_heads      = 8 
num_enc_layers = 3
num_dec_layers = 3
dropout        = 0.1
max_len        = 200 # used for positional embedding 
fwd_exp        = 4 
src_pad_idx    = english.vocab.stoi["<pad>"] 

t              = 0 # time step?

# need to configure loss plots here 
train_iter, validation_iter, test_iter = BucketIterator.splits(
  (train_data, validation_data, test_data),
  batch_size=batch_size,
  sort_within_batch=True,
  sort_key=lambda x: len(x.src),
  device=device
)

# instantiate model
model = Transformer(embedding_size, src_vocab_size, trg_vocab_size,
                    src_pad_idx, num_heads, num_enc_layers, num_dec_layers,
                    fwd_exp, dropout, max_len, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

easy_sentence = "ein pferd geht unter einer brücke neben einem boot um ein haus und durch eine schule."
easy_response = "a horse goes under a bridge next to a boat around a house and through a school."
# would need to increase the max_len for these longer sentences which drastically slows down training
hard_sentence = "Ein asiatisches Open-Air-Festival mit weißen Worten auf roten Spruchbändern, eine Bühne mit Richtern hinter eine Reihe roter Blumen, und viele asiatische Menschen, die herumstehen."
hard_response = "Outside asian festival with white words on red banners, a stage of judges behind a row of red flowers with lots of asian people standing by."
# Kafka seems fitting

# training loop 
for epoch in range(n_epochs):
  print(f"Epoch {epoch} / {n_epochs}")

  # save model each epoch maybe?
  model.eval()
  translated_sentence = translate_sentence(model, hard_sentence, german, english, device, max_len=max_len)

  cleaned_translation = ' '.join([s for s in translated_sentence]).replace(" ,", ",").replace(" .", ".").replace("<sos> ", "<sos>").replace(" <eos>", "<eos>")
  print(f"Generated Translation: {cleaned_translation}")
  print(f"Correct   Translation: {hard_response}")

  for step, batch in enumerate(train_iter):
    source = batch.src.to(device)
    target = batch.trg.to(device)

    output = model(source, target[:-1, :]) # get the last time step 

    output = output.reshape(-1, output.shape[2]) # dump the batch, the sentences in the batch, and the distribution over those words into a single tensor
    target = target[1:].reshape(-1) # remove sos
    optimizer.zero_grad()

    loss = criterion(output, target)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()

    # plot loss over all time steps - reduce frequency of plots 
    t = (epoch * len(train_iter)) + step
    if smooth_plots:
      if step % 50 == 0:
        plot_loss(step, loss, f'epoch {epoch}', 'Per Epoch', color=epoch_colors[epoch]) # loss for epoch
        plot_loss(t, loss, 'loss', algo_name)
    else: 
      plot_loss(step, loss, f'epoch {epoch}', 'Per Epoch', color=epoch_colors[epoch]) # loss for epoch
      plot_loss(t, loss, 'loss', algo_name)
      # print(f"{t:4d} Epoch {epoch}:{step:4d} loss: {loss:.3f}")

# TODO add the Bleu score func 