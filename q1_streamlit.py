import torch
import torch.nn as nn
import streamlit as st
import json

class NextChar(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x

with open('itos.json', 'r') as f:
    itos = json.load(f)

with open('stoi.json', 'r') as f:
    stoi = json.load(f)
    
stoi = {k: int(v) for k, v in stoi.items()}
itos = {int(k): v for k, v in itos.items()}

def tokenize_string(s):
    return [stoi[ch] for ch in s]

def generate(model, itos, stoi, block_size, input_string, max_len=50):
    input_string = input_string[-block_size:]
    # If the input string is less than block_size, pad it with '.'
    input_string = '.' * (block_size - len(input_string)) + input_string if len(input_string) < block_size else input_string
    context = tokenize_string(input_string)
    name = ''
    for i in range(max_len):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        if ch == '.':
            break
        name += ch
        context = context[1:] + [ix]
    return name


device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
emb_dim = 64
  
model = NextChar(20, len(stoi), emb_dim, 100).to(device)
model.load_state_dict(torch.load('model.pth'))

st.title('Shakespear Text Generator')

input_string = st.text_input('Enter the starting string', 'Thou shant not')
output_len = st.slider('Output length', 10, 200, 50)

if st.button('Generate'):
    output = generate(model, itos, stoi, 20, input_string, output_len)
    st.write(output)
