# tickets-agent

## Instalación
`pip install -r requirements.txt`

## Entrenamiento del modelo
`python training.py`

## Ejecución del chatbot
python chatbot.py

## Prompts
- Tags: Para solicitar tags, es importante agregar ":" después los siguientes labels:
__*tag, categoría, etiqueta; o sus plurales*__.
- Estados: Para solicitar estados, es importante agregar ":" después del label
__*estado o estdados*__.

## Prompts de ejemplo
- `¿Cuáles son los tickets con los tags: gato, test, gfa?`
- `¿Cuáles son los tickets con estado: Started?`
