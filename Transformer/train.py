import torch 
from model import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

SOS = 1
EOS = 2
MASK = 0
LEN_ALPHABET = 10

x = torch.tensor(
    [ [SOS, 5, 6, 4, 3, 9, 5, EOS, 0],
      [SOS,8, 7, 3, 4, 5, 6, 7, EOS] ]
    ).to(device)

trg = torch.tensor(
    [ [SOS, 7, 4, 3, 5, 9, EOS, 0], 
      [SOS, 5, 6, 2, 4, 7, 6, EOS] ]
    ).to(device)

src_pad_idx = MASK
trg_pad_idx = MASK

src_vocab_size = LEN_ALPHABET
trg_vocab_size = LEN_ALPHABET

model = Transformer(src_vocab_size, trg_vocab_size, 
                    src_pad_idx, trg_pad_idx,
                    device=device).to(device)

out = model(x, trg[:, :-1])
print(out.shape)
