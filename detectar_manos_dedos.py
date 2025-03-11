# ProyectoA - Detector de manos y dedos levantados en WebCam

import cv2
import mediapipe as medP

# Instanciamos la librería mediapipe con la IA para detectar los dedos levantados
mpManos = medP.solutions.hands
manos = mpManos.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDibujo = medP.solutions.drawing_utils

# Método para contar los dedos levantados en la imagen
def contarDedos(puntosReferencia, mano):
    puntaDedo = [mpManos.HandLandmark.THUMB_TIP, mpManos.HandLandmark.INDEX_FINGER_TIP,
               mpManos.HandLandmark.MIDDLE_FINGER_TIP, mpManos.HandLandmark.RING_FINGER_TIP,
               mpManos.HandLandmark.PINKY_TIP]
    puntoMedioDedo = [mpManos.HandLandmark.THUMB_IP, mpManos.HandLandmark.INDEX_FINGER_PIP,
               mpManos.HandLandmark.MIDDLE_FINGER_PIP, mpManos.HandLandmark.RING_FINGER_PIP,
               mpManos.HandLandmark.PINKY_PIP]

    dedosLevantados = 0
    # Mano derecha
    if mano == 0:
        if puntosReferencia.landmark[puntaDedo[0]].x < puntosReferencia.landmark[puntoMedioDedo[0]].x:
            dedosLevantados += 1
    else:
        # Mano izquierda
        if puntosReferencia.landmark[puntaDedo[0]].x > puntosReferencia.landmark[puntoMedioDedo[0]].x:
            dedosLevantados += 1

    for i in range(1, 5):
        if puntosReferencia.landmark[puntaDedo[i]].y < puntosReferencia.landmark[puntoMedioDedo[i]].y:
            dedosLevantados += 1

    return dedosLevantados

# Abrimos ventana con captura de vídeo de webcam del equipo
capturaVideo = cv2.VideoCapture(0)

while capturaVideo.isOpened():
    camaraIniciada, imagenCapturada = capturaVideo.read()
    if not camaraIniciada:
        print("No hay imagen en la webcam.")
        continue
    
    # Capturamos cada frame del vídeo para analizarlo
    imagenCapturada = cv2.cvtColor(cv2.flip(imagenCapturada, 1), cv2.COLOR_BGR2RGB)
    # Obtenemos las posibles manos detectadas
    resultado = manos.process(imagenCapturada)
    imagenCapturada = cv2.cvtColor(imagenCapturada, cv2.COLOR_RGB2BGR)
    # Si hay puntos de referencia (manos detectadas), los dibujamos en la imagen
    if resultado.multi_hand_landmarks:        
        for idx, puntosReferencia in enumerate(resultado.multi_hand_landmarks):
            mano = idx % 2
            # Dibujamos cada punto de referencia en la imagen capturada en vivo
            mpDibujo.draw_landmarks(imagenCapturada, puntosReferencia, mpManos.HAND_CONNECTIONS)
            dedosContados = contarDedos(puntosReferencia, mano)
            # Mostramos texto en la imagen con las manos y dedos detectados
            if mano == 0:
                mano = "der."
            else:
                mano = "izq."
            cv2.putText(imagenCapturada, f"Mano {mano}: {dedosContados}", (10, 50 + (30 * idx)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("ProyectoA - Detector de manos y contador de dedos", imagenCapturada)
    
    # Si se pulsa la tecla "s" se cierra el programa
    if cv2.waitKey(5) & 0xFF == ord('s'):
        break
    
# Liberamos los recursos cargados de la webcam
capturaVideo.release()
cv2.destroyAllWindows()