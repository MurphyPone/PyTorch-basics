import torch 
import torch.nn as nn
import torch.optim as optim 
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np 
import random 
import spacy 

from model import * 

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
  return translated_sentence[1:]


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
max_len        = 100 # used for positional embedding 
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

easy_sentence = "ein pferd geht unter einer brücke neben einem boot."
easy_response = "a horse is walking under a bridge next to a boat."

hard_sentence = "Er lag auf seinem panzerartig harten Rücken und sah, wenn er den Kopf ein wenig hob, seinen gewölbten, braunen, von bogenförmigen Versteifungen geteilten Bauch, auf dessen Höhe sich die Bettdecke, zum gänzlichen Niedergleiten bereit, kaum noch erhalten konnte."
hard_response = "He lay on his armor-hard back, and saw, when he lifted his head a little, his bulging, brown belly, which was separated by ark-shaped ridges, at whose summit the blanket, ready to glide down completely, could hardly maintain itself."

# training loop 
for epoch in range(n_epochs):
  print(f"Epoch {epoch} / {n_epochs}")

  # save model each epoch maybe?
  model.eval()
  translated_sentence = translate_sentence(model, easy_sentence, german, english, device, max_len=max_len)

  print(f"Translation: \n {translated_sentence}")

  for i, batch in enumerate(train_iter):
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

    # plot loss 
    print(f"loss: {loss}")

# this takes awhile

# score = bleu(test_data, model, german, english, device)
# print(f"Bleu score: {score*100:.2f}")