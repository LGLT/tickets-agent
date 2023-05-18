import random
import json
import pickle
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

# Cargar el archivo de intenciones en formato JSON
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

words = []  # Lista para almacenar todas las palabras encontradas
classes = []  # Lista para almacenar todas las clases de intenciones encontradas
documents = []
ignore_letters = ['?', '!', '.', ',']

# Procesar las intenciones y sus patrones asociados
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)  # Tokenizar el patrón en palabras individuales
        words.extend(word_list)  # Agregar las palabras a la lista de palabras
        documents.append((word_list, intent['tag']))  # Agregar el documento (patrón, etiqueta) a la lista de documentos
        if intent['tag'] not in classes:
            classes.append(intent['tag'])  # Agregar la etiqueta a la lista de clases si no existe aún

print(documents, '\n')

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]  # Lematizar las palabras y eliminar los caracteres ignorados
words = sorted(set(words))  # Ordenar y eliminar duplicados de la lista de palabras
print(words)

classes = sorted(set(classes))  # Ordenar y eliminar duplicados de la lista de clases

pickle.dump(words, open('model/words.pkl', 'wb'))  # Guardar la lista de palabras en un archivo pickle
pickle.dump(classes, open('model/classes.pkl', 'wb'))  # Guardar la lista de clases en un archivo pickle

training = []  # Lista para almacenar los datos de entrenamiento
output_empty = [0] * len(classes)  # Lista de ceros para generar las salidas codificadas

# Crear los datos de entrenamiento con la representación de bolsa de palabras y las salidas codificadas
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]  # Lematizar y convertir en minúsculas las palabras del patrón
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)  # Agregar 1 si la palabra está presente en el patrón, de lo contrario, agregar 0
    
    output_row = list(output_empty)  # Crear una copia de la lista de salidas codificadas
    output_row[classes.index(document[1])] = 1  # Establecer el índice correspondiente a la etiqueta en 1 para indicar la clase correcta
    training.append([bag, output_row])  # Agregar la representación de bolsa de palabras y la salida codificada a los datos de entrenamiento
    
random.shuffle(training)        # Mezclar los datos de entrenamiento de manera aleatoria
training = np.array(training)   # Convertir los datos de entrenamiento en un arreglo numpy

train_x = list(training[:, 0])  # Extraer las representaciones de bolsa de palabras como características de entrenamiento
train_y = list(training[:, 1])  # Extraer las salidas codificadas como etiquetas de entrenamiento

model = Sequential() # Crear un modelo secuencial de Keras
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) # Capa oculta densa con 128 neuronas y función de activación ReLU
model.add(Dropout(0.5)) # Agregar una capa de dropout para regularización
model.add(Dense(64, activation='relu')) # Capa oculta densa con 64 neuronas y función de activación ReLU
model.add(Dropout(0.5)) # Agregar otra capa de dropout para regularización
model.add(Dense(len(train_y[0]), activation='softmax')) # Capa de salida con activación softmax para clasificación multiclase

initial_learning_rate = 0.01 # Tasa de aprendizaje inicial
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True)
sgd = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

model.save('model/chatbotmodel.h5', hist)  # Guardar el modelo entrenado en un archivo h5
print('Done')