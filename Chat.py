import random
import json
import torch
from model import NeuralNet
from app import bag_of_words, tokenize
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intent.json','r',encoding='utf-8') as f:
    intents=json.load(f)
FILE = "data.pth"
data=torch.load(FILE)
input_size=data['input_size']
hidden_size=data['hidden_size']
output_size=data['output_size']
all_words=data['all_words']
model_state=data['model_state']
tags=data['tags']
model= NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()
bot_name="Alice"
input_1=input("To Start Chatting Write Down Your Name first: ")
print(f"Hi {input_1} Let's Chat: type 'quit' or 'q' to exit")
while True :
    sentence=input(f'{input_1}: ')
    if sentence in [x.lower() for x in ["Quit", "Exit", "Q"]]:
        break
    sentence=tokenize(sentence)
    x=bag_of_words(sentence,all_words)
    x=x.reshape(1,x.shape[0])
    x=torch.from_numpy(x)
    output =model(x)
    _,predict=torch.max(output,dim=1)
    tag=tags[predict.item()]
    probs=torch.softmax(output,dim=1)
    prob=probs[0][predict.item()]
    if prob.item() > 0.98 :
        for intent in intents["intents"]:
            if tag ==intent["tag"]:
                print(f"{bot_name}:{random.choice(intent['responses'])}")

    else:
        generic_sent=['Sorry, can not understand you','Please give me more info','Not sure what you are saying','Could you please rephrase the Statements']
        print(f"{bot_name}:{random.choice(generic_sent)}")
