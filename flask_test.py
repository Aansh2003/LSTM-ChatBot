from flask import Flask,render_template,request,render_template_string
import torch
from md import Classifier_net
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
import os


filehtml = 'templates/index.html'
 
def weather(city):
    url = "https://www.google.com/search?q="+"weather"+city
    html = requests.get(url).content
    soup = BeautifulSoup(html, 'html.parser')
    temp = soup.find('div', attrs={'class': 'BNeawe iBp4i AP7Wnd'}).text
    str = soup.find('div', attrs={'class': 'BNeawe tAd8D AP7Wnd'}).text
    data = str.split('\n')
    return data[1],data[0],temp


device = 'cpu'

FILE = "data.pth"
data = torch.load(FILE,map_location=device)

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

my_model = Classifier_net(inputs,output,hidden,1,1).to(device)
my_model.load_state_dict(state_model)
my_model.eval()

punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

with open(filehtml) as file:
    htmlFile = file.read()
    soup = BeautifulSoup(htmlFile,features="lxml")


def query(sentence):
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
    out = my_model(torch.LongTensor(val).to(device))
    out_cpu = out.cpu()
    idx = torch.argmax(out_cpu).numpy()
    prob = torch.softmax(out_cpu,dim=1)
	
    for intent in intents['intents']:
        if my_tags[idx] == intent['tag']:
            if torch.max(prob).detach().numpy() > 0.6:
                #print("Probability: "+str(torch.max(prob).detach().numpy()))
                output = random.choice(intent['responses'])
                if my_tags[idx] == "weather":
                    desc,my_time,temp = weather('manipal')
                    output = "It is"+ temp +", the sky is"+desc
            elif my_tags[idx] == "Emotion-good" or my_tags[idx] == "Emotion-bad" and torch.max(prob).detach().numpy() > 0.35:
                output = random.choice(intent['responses'])
            else:
                #print("Probability: "+str(torch.max(prob).detach().numpy()))
                #print("Max class: "+my_tags[idx])
                output = 'Sorry, I don\'t understand...\n'
    return output

head = soup.find("div", {"class": "chat-window"})

app = Flask(__name__)
@app.route('/')
def my_form():
    try:
        os.rmdir('templates/temp')
        os.mkdir('templates/temp')
    except:
         pass
    return render_template('index.html')

@app.route('/', methods=['POST','GET'])
def my_form_post():
    if request.method == 'POST':
        sentence = request.form['text']
        out = query(sentence)
        new_p = soup.new_tag("p",**{'class':'user'})
        new_p.string = sentence
        head.append(new_p)
        new_p2 = soup.new_tag("p",**{'class':'bot'})
        new_p2.string = out
        head.append(new_p2)
        print(head)
        # what = 'index_new'+'.html'
        # divTag = soup.new_tag('div') 
        # divTag['class'] = "link" 
        # headTag.insert_after(divTag)
        # with open("templates/temp/"+what,"w",encoding='utf-8') as my_file:
        #     my_file.write(str(soup))

        # print('what')

    return render_template_string(str(soup))


if __name__ == '__main__':
    app.run()

# <div class="message">
#     <p></p>
# </div>
# <div class="message">
#     <p></p>
# </div>