import numpy as np
import matplotlib.pyplot as plt
longueur = 8

x_entrer = np.array(([3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5, 0.5], [2, 0.5], [5.5, 1], [1, 1], [4, 1.5]),
                    dtype=float)  # données d'entrer
y = np.array(([1], [0], [1], [0], [1], [0], [1], [0]), dtype=float)  # données de sortie /  1 = rouge /  0 = bleu

liste_erreur = []

# Changement de l'échelle de nos valeurs pour être entre 0 et 1
x_entrer = x_entrer / np.amax(x_entrer, axis=0)  # On divise chaque entré par la valeur max des entrées

# On récupère ce qu'il nous intéresse
X = np.split(x_entrer, [longueur])[0]  # Données sur lesquelles on va s'entrainer, les 8 premières de notre matrice
xPrediction = np.split(x_entrer, [longueur])[1]  # Valeur que l'on veut trouver


# Notre classe de réseau neuronal
class Neural_Network(object):

    def __init__(self):

        # Nos paramètres
        self.inputSize = 2  # Nombre de neurones d'entrer
        self.outputSize = 1  # Nombre de neurones de sortie
        self.hiddenSize = 3  # Nombre de neurones cachés

        # Nos poids
        self.W1 = np.random.randn(self.inputSize,
                                  self.hiddenSize)  # (2x3) Matrice de poids entre les neurones d'entrer et cachés
        self.W2 = np.random.randn(self.hiddenSize,
                                  self.outputSize)  # (3x1) Matrice de poids entre les neurones cachés et sortie
        self.liste_erreur_round = []

    # Fonction de propagation avant
    def forward(self, X):

        self.z = np.dot(X, self.W1)  # Multiplication matricielle entre les valeurs d'entrer et les poids W1
        self.z2 = self.sigmoid(self.z)  # Application de la fonction d'activation (Sigmoid)
        self.z3 = np.dot(self.z2, self.W2)  # Multiplication matricielle entre les valeurs cachés et les poids W2
        o = self.sigmoid(
            self.z3)  # Application de la fonction d'activation, et obtention de notre valeur de sortie final
        return o

    # Fonction d'activation
    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    # Dérivée de la fonction d'activation
    def sigmoidPrime(self, s):
        return s * (1 - s)

    # Fonction de rétropropagation
    def backward(self, X, y, o):

        self.o_error = y - o  # Calcul de l'erreur
        self.liste_erreur_round = self.o_error

        self.o_delta = self.o_error * self.sigmoidPrime(o)  # Application de la dérivée de la sigmoid à cette erreur

        self.z2_error = self.o_delta.dot(self.W2.T)  # Calcul de l'erreur de nos neurones cachés
        self.z2_delta = self.z2_error * self.sigmoidPrime(
            self.z2)  # Application de la dérivée de la sigmoid à cette erreur

        self.W1 += X.T.dot(self.z2_delta)  # On ajuste nos poids W1
        self.W2 += self.z2.T.dot(self.o_delta)  # On ajuste nos poids W2
        return self.liste_erreur_round

    # Fonction d'entrainement
    def train(self, X, y):

        o = self.forward(X)
        liste_erreur_round = self.backward(X, y, o)
        return liste_erreur_round

    # Fonction de prédiction
    def predict(self):

        print("Donnée prédite apres entrainement: ")
        print("Entrée : \n" + str(arrondi(xPrediction)))
        print("Sortie : \n" + str(arrondi(self.forward(xPrediction))))

        if (self.forward(xPrediction) < 0.5):
            print("La fleur est BLEU ! \n")
        else:
            print("La fleur est ROUGE ! \n")

def arrondi(nombre):
    nombre = np.matrix.round(nombre, 2)
    return nombre


def dessin(liste_erreur):
    liste_entiers = []
    for i in range(len(liste_erreur)):
        liste_entiers.append(i)
    # Data for plotting
    t = liste_entiers
    s = liste_erreur

    fig, ax = plt.subplots()
    ax.plot(t, s)

    plt.ylim(0,1)

    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           title='About as simple as it gets, folks')
    ax.grid()

    fig.savefig("test.png")
    plt.show()


NN = Neural_Network()

#entrainement
for i in range(100):  # Choisissez un nombre d'itération, attention un trop grand nombre peut créer un overfitting !
    liste_erreur_round = NN.train(X, y)
    liste_erreur.append(sum(abs(liste_erreur_round))/len(liste_erreur_round))
    print("# " + str(i) + "\n")
    for j in range(longueur):
        print("Valeurs d'entrées: " + str(arrondi(X[j])), "Sortie actuelle: " + str(y[j]), "Sortie prédite: " +
              str(arrondi(NN.forward(X[j]))), 'erreur = ' + str(arrondi(liste_erreur_round[j])))
    print("\n")


NN.predict()
dessin(liste_erreur)
