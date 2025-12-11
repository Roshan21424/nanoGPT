# inspecting the dataset
with open('input.txt','r',encoding='utf-8') as file:
  dataset=file.read()
print("length of dataset: ",len(dataset)) # printing the length of dataset
print("first 1000 characters: ",dataset[:1000]) # printing the first 1000 characters from the dataset

# unique characters that occur in the dataset (vocabulory)
vocab = sorted(list(set(dataset)))
vocab_size = len(vocab)
print("vocab of dataset: ",vocab)
print("vocab size: ",vocab_size)

# encoding and decoding 
from re import I 
ctoi = {c:i for i,c in enumerate(vocab)} # {'\n':0,'a':1,'b':2,'c':3....}
itoc = {i:c for i,c in enumerate(vocab)} # {0:'/n',1:'a',2:'b',3:'c'....}

'''for each c in given text find its value from the ctoi '''
encode =lambda text:[ctoi[c] for c in text]

'''for each i in the encoding find its value from itoc and join them '''
decode = lambda encoded_text: ''.join([itoc[i] for i in encoded_text])

print(encode('hii there'))
print(decode(encode('hii there')))

import torch
data =torch.tensor(encode(dataset),dtype=torch.long)
print(data.shape,data.dtype)
print(data[:1000])

# spliting the data into training data and test data
n=int(0.9*len(data))
train_data=data[:n]
test_data=data[n:]

block_size=8
temp=train_data[:block_size+1]
print(temp)

x=train_data[:block_size]
y=train_data[1:block_size+1]
for i in range(block_size):
  context=x[:i+1]
  target=y[i]
  print(f"when input is {context} the target is {target}")

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel(batch dimension/b)
block_size = 8 # maximun context length of input prediction(time dimension/t)

def get_batch(split):
   # generate a small batch of data of inputs x and targets y
   data = train_data if split == 'train' else test_data
   ix = torch.randint(len(data)-block_size,(batch_size,)) # generate 4 random starting integers for the sequences
   x = torch.stack([data[i:i+block_size] for i in ix]) # create input batch 
   y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # create prediction batch
   return x,y

# xb is input batch
# yb is prediction batch
xb ,yb = get_batch('train')
print("input batch shape: ",xb.shape)
print(xb)
print("input batch shape: ",yb.shape)
print(yb)

for b in range(batch_size): 
  for t in range(block_size):
    context = xb[b,:t+1]
    target =yb[b,t]
    print(f"when input is {context.tolist()} the target is {target}")

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

  def __init__(self,vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

  def forward(self,xb,targets=None):
    ''' logits are scores for the next character in the sequence
        initially they are just random'''
    logits = self.token_embedding_table(xb)
    if targets is None:
      loss = None
    else:
      #reshape logits for cross entropy function
     B,T,C = logits.shape
     logits = logits.view(B*T,C)
     targets = targets.view(B*T)
     loss = F.cross_entropy(logits,targets)
    return logits,loss

  def generate(self,xb,max_new_tokens):
    # max_new_tokens is the no of new tokens we want to generate
    for _ in range(max_new_tokens):
     logits,loss = self(xb)
     logits = logits[:,-1,:] #get the last tokens predictions
     probs = F.softmax(logits,dim=1) # perform softmax for the last tokens predictions so that they add up to one
     xb_next = torch.multinomial(probs,num_samples=1)
     xb = torch.cat((xb,xb_next),dim=1) # append the new token to the existing input sequence
     ''' if initially the idx was [1,2,3] now it will be [1,2,3,4] where 4 is the new token.
     this longer sequence is fed into the model again on the next loop iteration. '''
    return xb

m= BigramLanguageModel(vocab_size)
logits,loss = m(xb,yb)
print(logits.shape)
print(loss)

idx = torch.zeros((1,1),dtype=torch.long)
print(decode(m.generate((idx),max_new_tokens=1000)[0].tolist()))


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)


# train the model
batch_size = 32
for steps in range(10000):
  xb,yb = get_batch('train')
  logits,loss = m(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step() #apply gradient decent

  print(loss.item())

print(decode(m.generate((idx),max_new_tokens=1000)[0].tolist()))