# Validação cruzada
# Porcentagem de exemplos utilizados para compor o conjunto de validação
ARG_VALIDATION_PERCENTAGE = 0.5
# Número de repetições para a validação cruzada por k-repetições
ARG_KFOLD_REPETITIONS = 5

# Algoritmos
# O numero de árvores no Random forest 
ARG_RANDOM_FOREST_TREES = 10
# Porcentagem de atributos a serem utilizadas em cada árvore
ARG_RANDOM_MAX_FEATURES = 0.5
# Porcentagem de exemplos a serem utilizadas em cada árvore
ARG_RANDOM_MAX_SAMPLES = 0.2

# Redes
ACTIVATION_FUNCTION = 'sig'
MAX_EPOCHS = 30
BIAS = 1
LEARNING_RATE = 0.03
HIDDEN_LAYERS_SIZE = 1
NEURONS_OUTPUT_LAYER = 1
NEURONS_PER_LAYER = 5