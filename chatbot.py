import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

from pivotaltracker_api.stories import StoryAPI

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

def get_response(intents_list, intents_json, data):
    if data[0] == 'estados':
        story_api = StoryAPI()  # Crear una instancia de la clase StoryAPI
        api_response = story_api.get_all_stories(data[1])
    elif data[0] == 'tags':
        story_api = StoryAPI()  # Crear una instancia de la clase StoryAPI
        api_response = story_api.get_all_stories(data[1])
    elif len(data) == 2:
        story_api = StoryAPI()  # Crear una instancia de la clase StoryAPI\
        story_id = data[0][1]
        story_state = data[1][1]
        api_response = ''
        story_api.update_story(story_id, story_state)
        
    tag = intents_list[0]['intent']  # Obtener la etiqueta de la intención principal
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])  # Seleccionar una respuesta aleatoria de la intención correspondiente
            break
        
    response_string = f"{result}\n"
    response_string += '\n'.join(str(item) for item in api_response)
    return response_string

print("¡Bot listo para ayudar!")

def extract_message_data(ints, message):
    if ints[0]['intent'] == 'actualización':
        id_index = message.find("id:")
        ticket_index = message.find("ticket:")
        estado_index = message.find("estado:")

        # Función auxiliar para obtener el valor después de un substring
        def extract_value(substring, start_index):
            if start_index != -1:
                value = message[start_index + len(substring):].split()[0]
                return value
            return None

        # Obtener los valores de "id" y "estado"
        id_value = extract_value("id:", id_index) or extract_value("ticket:", ticket_index)
        estado_value = extract_value("estado:", estado_index)

        return (('id', id_value), ('estado', estado_value))
    
    # Verificar si se solicita una categoría
    if 'tag:' in message or 'tags:' in message or 'categoria:' in message or 'categorias:' in message or 'categoría:' in message or 'categorías:' in message or 'etiqueta:' in message or 'etiquetas:' in message:
        # Obtener el índice de los dos puntos
        two_dots = message.index(':')
        # Obtener los tags separados por comas
        tags = message[two_dots + 1:].strip().split(',')
        # Eliminar espacios en blanco alrededor de cada tag
        tags = [tag.strip().replace('?', '') for tag in tags]
        return ('tags', tags)
        
    # Verificar si se solicita un estado
    if 'estado:' in message or 'estados:' in message:
        # Obtener el índice de los dos puntos
        two_dots = message.index(':')
        # Obtener los estados separados por comas
        estados = message[two_dots + 1:].strip().split(',')
        # Eliminar espacios en blanco alrededor de cada tag
        estados = [estado.strip() for estado in estados]
        return ('estados', estados)

    # Si no se detecta ni una categoría ni un estado válido, retornar None
    return (None, None)
    

while True:
    message = input("").lower()  # Leer el mensaje del usuario desde la entrada estándar
    ints = predict_class(message)  # Predecir la intención del mensaje en minúsculas
    print(ints)
    data = extract_message_data(ints, message)
    print(data)
    res = get_response(ints, intents, data)  # Obtener la respuesta correspondiente a la intención
    print(res)  # Imprimir la respuesta