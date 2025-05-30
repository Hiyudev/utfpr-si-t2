# Repetições totais
ARG_TOTAL_REPETITIONS = 100

# Validação cruzada
# Porcentagem de exemplos utilizados para compor o conjunto de validação
ARG_VALIDATION_PERCENTAGE = 0.5
# Número de repetições para a validação cruzada por k-repetições
ARG_KFOLD_REPETITIONS = 5

# Discretização
ARG_DISCRE_QPA = "Normal"
ARG_DISCRE_QPA_GROUPS = 5
ARG_DISCRE_PULSO = "Percentil"
ARG_DISCRE_PULSO_GROUPS = 5
ARG_DISCRE_RESPIRACAO = "Otsu"
ARG_DISCRE_RESPIRACAO_GROUPS = 5

# Algoritmos
# Criterio de divisao do Decision Tree
ARG_DECISION_TREE_CRITERION = "gini"

# Criterio de divisão do Random Forest
ARG_RANDOM_FOREST_CRITERION = "entropy"
# Número de estimadores no Random Forest
ARG_RANDOM_FOREST_ESTIMATORS = 100
# Porcentagem de atributos a serem utilizadas em cada árvore
ARG_RANDOM_MAX_FEATURES = 0.5
# Porcentagem de exemplos a serem utilizadas em cada árvore
ARG_RANDOM_MAX_EXAMPLES = 0.2
