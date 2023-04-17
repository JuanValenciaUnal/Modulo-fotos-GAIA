import cv2
import os

# Crea un objeto de captura de video
cap = cv2.VideoCapture(0)

# Crea la carpeta si no existe
if not os.path.exists('fotos'):
    os.makedirs('fotos')

while True:
    # Lee un fotograma de la cámara
    ret, frame = cap.read()

    # Muestra el fotograma en una ventana
    cv2.imshow('Camara', frame)

    # Espera a que se presione una tecla
    key = cv2.waitKey(1)

    # Si se presiona la tecla "q", toma una foto
    if key == ord('q'):
        # Guarda la foto en la carpeta "fotos"
        cv2.imwrite('fotos/imagen.png', frame)

    # Si se presiona la tecla "ESC", sale del bucle
    if key == 27:
        break

# Libera la cámara y cierra la ventana
cap.release()
cv2.destroyAllWindows()







