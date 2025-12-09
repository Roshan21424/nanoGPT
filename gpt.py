#downloading tiny shakesphere dataset
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

#inspecting the dataset
with open('input.txt','r',encoding='utf-8') as file:
  text=file.read()
print("length of dataset: ",len(text)) #printing the length of dataset
print(text[:1000]) #printing the first 1000 words

#unique characters that occur in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("unique chars in dataset: ",chars)
print("vocab size: ",vocab_size)

#character tokenization  
from re import I
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode =lambda s:[stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
print(encode('hii there'))
print(decode(encode('hii there')))