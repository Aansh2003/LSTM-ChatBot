import nltk
from nltk.stem.porter import PorterStemmer
import torch
from torch import nn

stemmer = PorterStemmer()

class Net(nn.Module):
    def __init__(self,input,output,hidden):
        super(Net,self).__init__()
        self.l1 = nn.Linear(in_features=input,out_features=hidden)
        self.l2 = nn.Linear(in_features=hidden,out_features=hidden)
        self.l3 = nn.Linear(in_features=hidden,out_features=output)
        self.relu = nn.LeakyReLU()

    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)

        return out
    
class Classifier_net(nn.ModuleList):
	def __init__(self, input,output,hidden,batch,lstm):
		super(Classifier_net, self).__init__()
		
		# Hyperparameters
		self.batch_size = batch
		self.hidden_dim = hidden
		self.LSTM_layers = lstm
		self.input_size = input
		self.output_size = output
		
		self.dropout = nn.Dropout(0.5)
		self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
		self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers, batch_first=True)
		self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim*2)
		self.fc2 = nn.Linear(self.hidden_dim*2, output)
		
	def forward(self, x):
		h = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim)).to('cuda')
		c = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim)).to('cuda')

		torch.nn.init.xavier_normal_(h)
		torch.nn.init.xavier_normal_(c)

		out = self.embedding(x)

		out, (hidden, cell) = self.lstm(out, (h,c))
		out = self.dropout(out)
		out = torch.relu_(self.fc1(out[:,-1,:]))
		out = self.dropout(out)
		out = self.fc2(out)

		return out
