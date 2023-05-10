import cv2
import os

# Función que determina si un contorno contiene texto
def has_text_inside(contour, gray):
    # Obtener las coordenadas y dimensiones del contorno
    x, y, w, h = cv2.boundingRect(contour)
    # Obtener la región de interés (ROI) en escala de grises
    roi = gray[y:y+h, x:x+w]
    # Aplicar umbralización para obtener la imagen binaria de la ROI
    _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Contar los píxeles blancos en la imagen binaria
    n_white_pix = cv2.countNonZero(roi_thresh)
    # Determinar si hay suficientes píxeles blancos para considerar que hay texto
    return n_white_pix > 0.2 * w * h

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Crear la carpeta para guardar las imágenes capturadas si no existe
if not os.path.exists('fotos'):
    os.makedirs('fotos')

while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Aplicar un filtro Gaussiano para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detectar bordes con el algoritmo Canny
    edges = cv2.Canny(blurred, 50, 200)
    # Encontrar contornos en la imagen de bordes
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Hacer una copia del frame para dibujar sobre él
    img_copy = frame.copy()
    # Dibujar un rectángulo verde en el centro del frame
    cv2.rectangle(img_copy, (240, 160), (400, 320), (0, 255, 0), 2)
    # Iterar sobre todos los contornos encontrados
    for contour in contours:
        # Descartar los contornos muy pequeños
        if cv2.contourArea(contour) < 2500:
            continue
        # Descartar los contornos que no contienen texto
        if not has_text_inside(contour, gray):
            continue
        # Obtener las coordenadas y dimensiones del contorno
        x, y, w, h = cv2.boundingRect(contour)
        # Si el contorno está dentro del rectángulo verde, guardar la imagen
        if 240 < x < 400 and 160 < y < 320:
            cv2.imwrite('fotos/imagen.png', frame[y:y+h, x:x+w])
    # Mostrar el frame con el rectángulo verde
    cv2.imshow('Camara', img_copy)
    # Esperar una tecla durante 1 milisegundo
    key = cv2.waitKey(1)
    # Si se presiona la tecla ESC, salir del bucle
    if key == 27:
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
