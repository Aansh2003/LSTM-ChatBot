import md
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import torch
from torch import nn

import json

stemmer = PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

with open('intents.json','r') as f:
    intents = json.load(f)

all_words = []
tags = []
two_d = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        two_d.append((pattern,tag))

to_ignore = [',','.','','-','?','!']
all_words = [stem(a) for a in all_words if a not in to_ignore]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim
from keras_preprocessing.sequence import pad_sequences

class Preprocessing:
	def __init__(self):
		self.max_words = len(all_words)
		
	def load_data(self):
		self.x = x_train
		self.y = y_train
		
	def prepare_tokens(self):
		self.tokens = Tokenizer(num_words=self.max_words)
		self.tokens.fit_on_texts(self.x)

	def sequence_to_token(self, x):
		sequences = self.tokens.texts_to_sequences(x)
		return pad_sequences(sequences, maxlen=20)

class Loader:
    def __init__(self,x_train,y_train):
        self.samples = len(x_train)
        self.x = x_train
        self.y = y_train

    def __getitem__(self,index):
        return self.x[index],self.y[index]

    def __len__(self):
        return self.samples
    
from sklearn.metrics import accuracy_score

class Execute:
	def __init__(self, input,output,hidden,batch_size,lstm_layers):
		self.__init_data__()
		self.batch_size = batch_size
		
		self.model = md.Classifier_net(input,output,hidden,1,1).to('cuda')
		
	def __init_data__(self):
		self.preprocessing = Preprocessing()
		self.preprocessing.load_data()
		self.preprocessing.prepare_tokens()

		raw_x_train = self.preprocessing.x
		self.y_train = self.preprocessing.y

		self.x_train = self.preprocessing.sequence_to_token(raw_x_train)
		
	def train(self):
		
		training_set = Loader(self.x_train, self.y_train)		
		self.loader_training = DataLoader(training_set, batch_size=self.batch_size)
		criterion = torch.nn.CrossEntropyLoss().to('cuda')
		optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.001)
		for epoch in range(20000):
			predictions = []
			self.model.train()
			for x_batch, y_batch in self.loader_training:
				
				x = x_batch.type(torch.LongTensor).to('cuda')
				y = y_batch.type(torch.LongTensor).to('cuda')
				y_pred = self.model(x)
				loss = criterion(y_pred,y)
				optimizer.zero_grad()
				
				loss.backward()
				
				optimizer.step()
				
				predictions += list(y_pred.cpu().squeeze().detach().numpy())
		
		self.model = self.model.eval()
		out = self.model(x_batch.type(torch.LongTensor).to('cuda'))
		count = 0
		out_cpu = out.cpu()
		for i in range(len(y_pred)):
			if torch.argmax(out_cpu[i]).numpy() == y_batch.type(torch.LongTensor)[i].numpy():
				count += 1
		print(count/len(y_pred))

if __name__ == "__main__":
    x_train = []
    y_train = []

    for pattern_sent,tag_name in two_d:
        x_train.append(pattern_sent)
        label = tags.index(tag_name)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    batch_size = len(all_words)
    hidden = 8
    output = len(tags)
    input = len(all_words)
    lstm_layers = 1
    learning_rate = 0.0001
    epochs = 10000


    exec = Execute(input,output,hidden,batch_size,lstm_layers)
    exec.train()

    data = {
        "model_state":exec.model.state_dict(),
        "input_size":input,
        "output_size":output,
        "hidden_size":hidden,
        "words":all_words,
        "tags":tags,
        "preprocess":exec.preprocessing
    }

    FILE = "data.pth"
    torch.save(data,FILE)
