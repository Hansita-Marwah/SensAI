import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utility import bag_of_words, tokenize, stem
from my_model import NeuralNet

with open('intents.json', 'r') as l:
    intents = json.load(l)

all_words = []
tags = []
all = []

for intent in intents['intents']:
    tag = intent['tag']
   
    tags.append(tag)
    for pattern in intent['patterns']:
        a = tokenize(pattern)
        all_words.extend(w)
        all.append((w, tag))
ignore_words = ['?', '.', '!']
all_words = [stem(a) for a in all_words if a not in ignore_words]

all_words = sorted(set(all_words))
tags = sorted(set(tags))

#print(len(all), "patterns")
#print(len(tags), "tags:", tags)
#print(len(all_words), all_words)

X_way = []
y_way = []
for (pattern_sentence, tag) in xy:
   
    bag = bag_of_words(pattern_sentence, all_words)
    X_way.append(bag)
    
    label = tags.index(tag)
    y_way.append(label)

X_way = np.array(X_way)
y_way = np.array(y_way)

num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_way[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_way)
        self.x_data = X_way
        self.y_data = y_way

    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

 
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
    
        outputs = model(words)
       
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
