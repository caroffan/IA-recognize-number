from keras.datasets import mnist
from matplotlib import pyplot
import random
import numpy
from numpy.core.numeric import zeros_like
(train_X, train_y), (test_X, test_y) = mnist.load_data()

TrainY = []
for i in range(len(train_y)):
    tab = numpy.zeros(10)
    tab[train_y[i]] = 1
    TrainY.append(tab)

#paramètre du réseau
Nlayer =20 #nombre de neurone de la couche intermédiaire
Nsortie = 10 #nombre de neurone de sortie
Nentree = 784 #nombre de neurone de entrée
u = 0.1 #taux d'apprentissage définit arbitrerement
nbIt = 100000 #nombre d'itération pour l'apprentissage

biais1 = numpy.random.rand(Nlayer)/10 #nombre de biais de la couche intermédiaire
poids1 = numpy.random.rand(Nentree,Nlayer)/10 #nombre de poids de la couche intermédiaire
biais2 = numpy.random.rand(Nsortie)/10 #nombre de biais de la couche de sortie
poids2 = numpy.random.rand(Nlayer,Nsortie)/10 #nombre de poids de la couche de sortie


#fonction permettant de calculer les yi
# x : entrée 
# poids : poids de la couche suivante
#biais : biais de la couche suivante
def y(x, poids, biais):
    return numpy.dot(x, poids)+biais

#fonction permettant de calculer les sigmoides
# y : yi correspondant  
def sigm(y):
    return 1/(1+numpy.exp(-y))


#fonction qui transforme un input X en son output à travers le reseau
def output(x, poids1, biais1, poids2, biais2):
    y1 = y(x, poids1, biais1)
    z1 = sigm(y1)
    y2 = y(z1, poids2, biais2)
    z2 = sigm(y2)
    return z2


for i in range(nbIt):
    n = random.randint(0, len(train_X)-1)
    x = [numpy.ravel(train_X[n])/255]#on prend l'image qu'on transforme en  1 tableau de 1 seul ligne et on divise par 255 pour avoir des valeurs entre 0 et  1, on met se tableau dans un tableau pour que sa taille soit compatible au calcule de matrice des poids
    Z2 = output(x, poids1, biais1, poids2, biais2) #on applique la fonction qui transforme notre input à travers le reseau
    
    #-----------------------------------------------------------------------------------------------------------------------------------------
    response = 0
    for i in range(len(Z2[0])):
        if round(Z2[0][i])!=0:
            response = i

    C = (Z2-TrainY[n])**2 # fonction de cout, Z2 est notre sortie, TrainY[n] est notre sortie théorique

    Cz2 = 2*(Z2 - TrainY[n])*Z2*(1-Z2)# bloque commun à toute les equations de descente de gradient
    Cz2 = Cz2.T #on prend la transposé car pour les calculs matriciels suivant nous devont inversé la taille de la matrice
    derivCb2 = Cz2 #descente de gradient pour les Biais de la deuxieme couche
    derivCw2 = numpy.dot(Cz2,sigm(y(x, poids1, biais1)))#descente de gradient pour les Poids de la deuxieme couche
    derivCb1 = numpy.dot(Cz2.T,poids2.T)*(sigm(y(x, poids1, biais1))*(1-sigm(y(x, poids1, biais1))))#descente de gradient pour les Biais de la première couche
    derivCw1 = numpy.dot(derivCb1.T, x)#descente de gradient pour les poids de la première couche

    poids1 = poids1 - u*derivCw1.T #maj des valeurs de poids de la premiere couche
    biais1 = biais1 - u*derivCb1 #maj des valeurs de poids de la premiere couche
    poids2 = poids2 - u*derivCw2.T #maj des valeurs de poids de la premiere couche
    biais2 = biais2.T - u*derivCb2.T #maj des valeurs de poids de la premiere couche 
    biais2 = biais2[0] #je remet les tableaux des biais sous la meme forme que celle de départ
    biais1 = biais1[0]

print("######################################################    RESULTAT    ###################################################################")
print("Sortie attendu : ", end="")
print(TrainY[n], end=" = ") #on affiche notre output
print(train_y[n])
print("-----------------------------------------------------------------------------------------------------------------------------------------")
response = 0
for i in range(len(Z2[0])):
    if round(Z2[0][i])!=0:
        response = i
print("Sortie : ", end="")
print(Z2, end=" = ") #on affiche notre output
print(response) #on affiche notre output


#on fait un test avec une image aleatoire test qu'il n'a jamais vu dans l'entrainement
tabX = []
RepX = []
RepAttend = []
fig = pyplot.figure(figsize=(10, 1))
for i in range(10):
    n = random.randint(0, len(test_X)-1)
    RepAttend.append(test_y[n])
    x = [numpy.ravel(test_X[n])/255]
    Z2 = output(x, poids1, biais1, poids2, biais2)
    tabX.append(x)
    response = 0
    for i in range(len(Z2[0])):
        if round(Z2[0][i])!=0:
            response = i
    RepX.append(response)
    fig.add_subplot(1, 10, i+1).axis('off')
    pyplot.imshow(test_X[n], cmap=pyplot.get_cmap('gray'))
print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("RESULTAT THEORIQUE TEST AVEC UNE AUTRE IMAGE DE MEME VALEUR : ", end="")
print(RepAttend)
print("RESULTAT TEST AVEC UNE AUTRE IMAGE DE MEME VALEUR :           ", end="")

print(RepX)

    
pyplot.show() 