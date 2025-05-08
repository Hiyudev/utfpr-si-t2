import sys

from libs.id3 import algorithm_id3
from libs.input import Exemplo, read_data

def id3(input: list[Exemplo]) -> list[str]:
    """
    Algoritmo ID3 para construção de árvores de decisão.
    """
    
    # Cria a árvore de decisão
    # tree = algorithm_id3(input)
    
    pass

def forest(input: list[Exemplo]) -> list[str]:
    pass

def redes(input: list[Exemplo]) -> list[str]:
    pass

def analyse(algorithm: function, validation: str):
    """
    Função para analisar o desempenho do algoritmo.
    """
    examples = read_data("./assets/treino_sinais_vitais_com_label.txt")
    examples_length = len(examples)
    
    if validation == "retencao":
        # Porcentagem de dados utilizados para o conjunto de validação, o restante será utilizado para o treinamento
        ARG_VALIDATION_PERCENTAGE = 0.2
        
        training_set: list[Exemplo] = []
        validation_set: list[Exemplo] = []

        for i in range(examples_length):
            # Escolhe um exemplo aleatório para o conjunto de validação
            random_example = examples.pop(examples.index(examples[i]))
            
            # Adiciona ao conjunto de validação se a porcentagem de validação não for atingida
            if len(validation_set) < examples_length * ARG_VALIDATION_PERCENTAGE:
                validation_set.append(random_example)
            else:
                training_set.append(random_example)
                
        # Executa o algoritmo com o conjunto de treinamento
        algorithm(training_set)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        algorithm: function = None
        
        algorithm_arg, validation = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None
        if algorithm_arg == "id3":
            algorithm = id3
        elif algorithm_arg == "forest":
            algorithm = forest
        elif algorithm_arg == "redes":
            algorithm = redes
        else:
            print("Argumento inválido. Use 'id3', 'forest' ou 'redes'.")
    
        if algorithm:
            analyse(algorithm, validation)
    else:
        print("Argumento inválido. Use 'id3', 'forest' ou 'redes'.")
