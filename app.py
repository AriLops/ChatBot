import json

# carregar arquivo intents.json
with open('intents.json', 'r', encoding='utf8') as file:
    intents = json.load(file)

import tensorflow as tf

# Carregar o conjunto de dados de treinamento e teste
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocessamento dos dados
X_train = X_train.reshape(-1, 784).astype('float32') / 255.
X_test = X_test.reshape(-1, 784).astype('float32') / 255.
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Definir o modelo da rede neural
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_dim=784),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

# Crie o objeto ChatBot
chatbot = ChatBot('MeuChatBot')

# Treine o ChatBot com dados da linguagem natural
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train('chatterbot.corpus.portuguese')

# Função de conversação
def conversar():
    while True:
        try:
            user_input = input('Você: ')
            response = chatbot.get_response(user_input)
            print('ChatBot: ', response)

        except (KeyboardInterrupt, EOFError, SystemExit):
            break
import requests
from bs4 import BeautifulSoup

# Faz uma requisição HTTP para a página web desejada
page = requests.get("https://www.google.com")

# Extrai o conteúdo HTML da página web
soup = BeautifulSoup(page.content, 'html.parser')

# Encontra todos os elementos de texto da página web
text_elements = soup.find_all('p')

# Imprime o conteúdo dos elementos encontrados
for element in text_elements:
    print(element.text)

    import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyNet, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.5)
        
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

        import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

        import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

        import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.form['user_message']  # Obtém a mensagem do usuário do formulário HTML
    response = ''

    # Lógica do chatbot - aqui você pode adicionar a lógica que desejar para responder ao usuário
    if user_message == 'Olá':
        response = 'Olá, tudo bem?'
    elif user_message == 'Qual é o seu nome?':
        response = 'Meu nome é ChatBot.'
    else:
        response = 'Desculpe, não entendi o que você quis dizer.'

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run()

    import json

with open('intents.json') as file:
    intents = json.load(file)

# Agora você pode acessar as intenções, padrões e respostas definidas no arquivo JSON assim:
for intent in intents['intents']:
    print(intent['tag'])
    for pattern in intent['patterns']:
        print(pattern)
    for response in intent['responses']:
        print(response)