import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy import random

img_cuatro = cv2.imread('cuatro.png')
#plt.imshow(img_cuatro)
img = cv2.imread('digits.png')
#plt.imshow(img)

# Preprocesamiento : En el preprocesamiento vamos a extraer los parches de 20x20 pixeles que se encuentran en la base de datos.
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Dividimos la imagen en 5 mil (50x100) parches de 20x20 cada uno
parches = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

# Conversion a np.array de tamaño (50,100,20,20)
x = np.array(parches)

#Visualicemos uno de esos parches
print(x[1,1,:,:])
plt.imshow(x[1,1,:,:])

#Dividir la base de datos
#Como vimos tenemos disponibles 5 mil ejemplos. Vamos a dividir estos ejemplos en dos conjuntos:

#Entrenamiento. Conjunto que utilizara el algoritmo para buscar los vecinos mas cercanos. Recordemos que este conjunto debe estar previamente etiquetado.
#Prueba. Conjunto donde probaremos que tan bueno seran las predicciones. Puedo o no estar etiquetado. En este caso sabemos las etiquetas

# Dividiremos en partes iguales
entrenamiento = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
prueba = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

print(len(entrenamiento))
print(len(prueba))

# Create labels for train and test data
etiquetas = np.arange(10)
print('Etiquetas:', etiquetas)

train_labels = np.repeat(etiquetas,250)[:,np.newaxis]
test_labels = train_labels.copy()

print(len(train_labels))
print(train_labels)

#Entrenamiento
#Formalmente debe haber un entrenamiento. En este caso, K-NN empareja cada uno de los ejemplos dados con sus respectivas etiquetas.

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.ml.KNearest_create()
knn.train(entrenamiento, cv2.ml.ROW_SAMPLE, train_labels)

# Prediccion
# Ahora probaremos el algoritmos para los ejemplos que dejamos aparte.
k_vecinos = 5

ret,result,neighbours,dist = knn.findNearest(prueba,k_vecinos)

print('Numero de predicciones:', len(result))
print(result)

# Some random examples
indice = random.randint(2500)
img_test = prueba[indice].reshape(20,20)
print('Etiqueta predicha:', result[indice])
print('Vecinos cercanos:', neighbours[indice])
    
plt.imshow(img_test)

# Evaluacion de los resultados
#Para saber que también esta funcionando el algoritmo compararemos las predicciones con las etiquetas reales, las cuales sabemos de antemano por su posición en la base de datos.

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
print("Número de predicciones correctas:", correct)
accuracy = correct*100.0/result.size
print ('Precision:', accuracy)