import cv2
import os

# Función para verificar si hay caracteres dentro del contorno
def has_text_inside(contour, gray):
    x, y, w, h = cv2.boundingRect(contour)
    roi = gray[y:y+h, x:x+w]
    _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n_white_pix = cv2.countNonZero(roi_thresh)
    return n_white_pix > 0.2 * w * h

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

    # Copia la imagen original
    img_copy = frame.copy()

    # Dibuja los contornos en la copia de la imagen
    cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 3)

    # Comprueba si alguno de los contornos tiene un tamaño mínimo y caracteres dentro de su área
    for contour in contours:
        if cv2.contourArea(contour) < 2500:
            continue
        if not has_text_inside(contour, gray):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        # Toma una foto y guárdala en la carpeta "fotos"
        cv2.imwrite('fotos/imagen.png', frame[y:y+h, x:x+w])

    # Muestra la copia de la imagen en una ventana
    cv2.imshow('Camara', img_copy)

    # Espera a que se presione una tecla
    key = cv2.waitKey(1)

    # Si se presiona la tecla "ESC", sale del bucle
    if key == 27:
        break

# Libera la cámara y cierra la ventana
cap.release()
cv2.destroyAllWindows()
