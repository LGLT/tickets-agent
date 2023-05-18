import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Cargar el archivo de intenciones en formato JSON
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

words = pickle.load(open('model/words.pkl', 'rb'))  # Cargar la lista de palabras desde el archivo pickle
classes = pickle.load(open('model/classes.pkl', 'rb'))  # Cargar la lista de clases desde el archivo pickle
model = load_model('model/chatbotmodel.h5')  # Cargar el modelo entrenado desde el archivo h5

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)  # Tokenizar la oración en palabras individuales
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]  # Lematizar las palabras
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)  # Obtener las palabras limpias de la oración
    bag = [0] * len(words)  # Crear un vector de ceros de longitud igual al número de palabras conocidas
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1  # Establecer el valor en 1 si la palabra está presente en la oración
    return np.array(bag)  # Convertir el vector de bolsa de palabras en un arreglo numpy

def predict_class(sentence):
    bow = bag_of_words(sentence)  # Obtener la representación de bolsa de palabras de la oración
    res = model.predict(np.array([bow]))[0]  # Predecir la clase de la oración utilizando el modelo
    ERROR_THRESHOLD = 0.25  # Umbral para filtrar las predicciones por debajo de cierta probabilidad
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]  # Obtener las predicciones que superan el umbral
    
    results.sort(key=lambda x: x[1], reverse=True)  # Ordenar las predicciones por probabilidad descendente
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})  # Crear una lista de intenciones y sus probabilidades
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']  # Obtener la etiqueta de la intención principal
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])  # Seleccionar una respuesta aleatoria de la intención correspondiente
            break
    return result

print("GO! Bot is running!")

while True:
    message = input("")  # Leer el mensaje del usuario desde la entrada estándar
    ints = predict_class(message.lower())  # Predecir la intención del mensaje en minúsculas
    res = get_response(ints, intents)  # Obtener la respuesta correspondiente a la intención
    print(res)  # Imprimir la respuesta