import cv2
import numpy as np
import face_recognition as fr
import os
import random
from datetime import datetime

#Acceso a la carpeta:
path = 'Fotos'
images =[]
clases = []
lista = os.listdir(path)
#print(lista)

#Variables: 
comp1 = 100

#Leer rostros de la "BD":

for lis in lista: 
    imgdb = cv2.imread(f'{path}/{lis}') #Lectura de las imagenes de los rostros.
    images.append(imgdb) #Almacenar imagen
    clases.append(os.path.splitext(lis)[0]) #Almacena por nombres

print(clases)

def codrostros(images):
    listacod = []

    #Iteracciones
    for img in images:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Correcion de color

        cod = fr.face_encodings(img)[0] #Codificacion de imagenes

        listacod.append(cod) #Almacenamos
    return listacod

def horario(nombre):

    with open ('Horario.csv', 'r+') as h: #Abrir el archivo en modo lectura y escritura
        data = h.readline() #Se lee la información
        listanombres = [] #Se crea la lista de nombre

        for line in data: #Iteracion cada linea del documento
            entrada = line.split(',')#Se busca la entrada
            listanombres.append(entrada[0])#Se almacenan los nombres

        if nombre not in listanombres: #Se verifica si ya se ha almacenado el nombre
            info = datetime.now() #Extraccion de informacion actual
            fecha = info.strftime('%y:%m:%D')
            hora = info.strftime('%H:%H:%S')

            h.writelines(f'\n{nombre},{fecha},{hora}')
            print (info)


rostroscod = codrostros(images) 

cap = cv2.VideoCapture(0) #Realizar videocaptura

while True:

    ret, frame = cap.read() #Lectura de fotogramas

    frame2 = cv2.resize(frame, (0,0), None, 0.25, 0.25) #Reduccion de imagenes

    rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB) #Conversion de color

    #Busquedad de rostros:
    faces = fr.face_locations(rgb) 
    facescod = fr.face_encodings(rgb, faces)

    #Iterraciones:
    for facecod, faceloc in zip(facescod, faces):
        
        comparacion = fr.compare_faces(rostroscod, facecod) #Comparacion rostro de DB con rostro en tiempo real
        simi = fr.face_distance(rostroscod, facecod) #Calcular la similitud
        min = np.argmin(simi)#Busquedad de valora mas bajo

        if comparacion[min]:
            nombre = clases[min].upper()
            print(nombre)
            yi, xf, yf, xi = faceloc #Extraccion de coordenadas
            yi, xf, yf, xi = yi*4, xf*4, yf*4, xi*4

            indice = comparacion.index(True)

            if comp1 != indice:
                r = random.randrange(0, 255, 50)
                g = random.randrange(0, 255, 50)
                b = random.randrange(0, 255, 50)

                comp1 = indice

            if comp1 == indice:
                cv2.rectangle(frame, (xi,yi), (xf,yf), (r,g,b), 3)
                cv2.rectangle(frame, (xi, yf-35), (xf, yf), (r,g,b), cv2.FILLED)
                cv2.putText(frame, nombre, (xi+6, yf-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                horario(nombre)

    #mostrar frames:
    cv2.imshow("Reconocimiento facial", frame)
    t = cv2.waitKey(5)
    if t == 27:
        break


cv2.destroyAllWindows()
cap.release()



