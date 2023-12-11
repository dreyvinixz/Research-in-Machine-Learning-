import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

random.seed(0)
n = 1000
azul = []
laranja = []
rótulo = []
for i in range(n):
    x = random.uniform(0, 1)
    y = random.uniform(0, 1)
    if x * x + y * y < 0.6:
        azul.append([x, y])
    else:
        laranja.append([x, y])

# gráfico de dispersão
plt.figure(figsize=(3, 3))
plt.plot([c[0] for c in azul],    [c[1] for c in azul], '.')
plt.plot([c[0] for c in laranja], [c[1] for c in laranja], '.');

# matriz de atributos
atributos = np.array(azul + laranja)
atributos.shape

# vetor de rótulos
rótulos = np.array(len(azul) * ['azul'] + len(laranja) * ['laranja'])
rótulos.shape

at_treino, at_teste, ro_treino, ro_teste = train_test_split(atributos, rótulos,
                                                            test_size=0.25,
                                                            random_state=0)

print(at_treino.shape, at_teste.shape, ro_treino.shape, ro_teste.shape)

classificador = DecisionTreeClassifier(random_state=0)

#from sklearn.svm import SVC
#classificador = SVC(random_state=0)

# treinamento
classificador.fit(at_treino, ro_treino)

# geração de previsões
print(classificador.predict([[0.7, 0.8]]))

# geração de todas as previsões
ro_teste_pred = classificador.predict(at_teste)

# medida de acurácia
from sklearn.metrics import accuracy_score
score = accuracy_score(ro_teste, ro_teste_pred)
print('{:.2%}'.format(score))

if type(classificador) == DecisionTreeClassifier:
    plt.figure(figsize=(8,4))
    plot_tree(classificador, feature_names=['x','y'], impurity=False, max_depth=3, proportion=True);