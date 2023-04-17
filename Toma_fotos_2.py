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

    # Convierte la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplica un filtro Gaussiano para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detecta los bordes en la imagen utilizando el operador de Canny
    edges = cv2.Canny(blurred, 50, 200)

    # Encuentra los contornos en la imagen
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Dibuja los contornos en la imagen original
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # Comprueba si alguno de los contornos tiene un tamaño mínimo
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:
            # Toma una foto y guárdala en la carpeta "fotos"
            cv2.imwrite('fotos/imagen.png', frame)

    # Muestra la imagen en una ventana
    cv2.imshow('Camara', frame)

    # Espera a que se presione una tecla
    key = cv2.waitKey(1)

    # Si se presiona la tecla "ESC", sale del bucle
    if key == 27:
        break

# Libera la cámara y cierra la ventana
cap.release()
cv2.destroyAllWindows()