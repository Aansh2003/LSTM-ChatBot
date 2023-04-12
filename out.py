import torch
import md
import sys
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from final import Preprocessing
import json
import final
import random
from bs4 import BeautifulSoup
import requests
import time
import jamspell
 
def weather(city):
    url = "https://www.google.com/search?q="+"weather"+city
    html = requests.get(url).content
    soup = BeautifulSoup(html, 'html.parser')
    temp = soup.find('div', attrs={'class': 'BNeawe iBp4i AP7Wnd'}).text
    str = soup.find('div', attrs={'class': 'BNeawe tAd8D AP7Wnd'}).text
    data = str.split('\n')
    print("Description: "+data[1])
    print("Time: "+data[0])
    print("Temperature: "+temp)

FILE = "data.pth"
data = torch.load(FILE,map_location='cpu')

corrector = jamspell.TSpellCorrector()
corrector.LoadLangModel('en.bin')

process = Preprocessing()

inputs = data['input_size']
output = data['output_size']
hidden = data['hidden_size']
words = data['words']
state_model = data['model_state']
my_tags = data['tags']

with open('intents.json','r') as f:
	intents = json.load(f)

two_d = []
tags = []

for intent in intents['intents']:
	tag = intent['tag']
	tags.append(tag)
	for pattern in intent['patterns']:
		two_d.append((pattern,tag))

x_train = []

for pattern_sent,tag_name in two_d:
	x_train.append(pattern_sent)

x_train = np.array(x_train)

tokens = Tokenizer(num_words=len(words))
tokens.fit_on_texts(x_train)

my_model = md.Classifier_net(inputs,output,hidden,1,1).to('cuda')
my_model.load_state_dict(state_model)
my_model.eval()

punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

print("Started...")
bot = "AI: "

while True:
	sentence = input("User: ")
	print('')
	sentence = sentence.lower()
	sentence = corrector.FixFragment(sentence)

	for ele in sentence:
		if ele in punc:
			sentence = sentence.replace(ele, "")
		if sentence == "":
			continue
	

	arr = []
	arr.append(sentence)
	sequence = tokens.texts_to_sequences(arr)
	val = pad_sequences(sequence, maxlen=20)
	out = my_model(torch.LongTensor(val).to('cuda'))
	out_cpu = out.cpu()
	idx = torch.argmax(out_cpu).numpy()
	prob = torch.softmax(out_cpu,dim=1)

	
	for intent in intents['intents']:
		if my_tags[idx] == intent['tag']:
			if torch.max(prob).detach().numpy() > 0.6:
				#print("Probability: "+str(torch.max(prob).detach().numpy()))
				time.sleep(1)
				print(bot+random.choice(intent['responses'])+'\n')
				if my_tags[idx] == "weather":
					weather('manipal')
			elif my_tags[idx] == "Emotion-good" or my_tags[idx] == "Emotion-bad" and torch.max(prob).detach().numpy() > 0.35:
				time.sleep(1)
				print(bot+random.choice(intent['responses']))
			else:
				#print("Probability: "+str(torch.max(prob).detach().numpy()))
				#print("Max class: "+my_tags[idx])
				print('Sorry, I don\'t understand...\n')

