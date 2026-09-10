"""
Paso 1: Balanza de Inclinación y Margen de Desplazamiento de Cabeza
"""
from pathlib import Path
import time
import cv2
import numpy as np

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
from mediapipe import Image, ImageFormat

# Rutas
ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
model_file = list(MODELS_DIR.glob("*face_landmarker*.task"))[0]
model_bytes = model_file.read_bytes()

# Landmarks clave para la cabeza
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
LEFT_FACE = 234
RIGHT_FACE = 454


def get_p2(lm, width, height):
    return np.array([int(lm.x * width), int(lm.y * height)])


def main():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_bytes),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
    )

    landmarker = FaceLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)

    # Anclaje de cabeza
    anchor_center = None
    max_allowed_drift = 50  # Máximo desplazamiento permitido en píxeles

    start_time = time.time()

    print("\n--- PASO 1: INDICADOR DE BALANZA ---")
    print("Mueve la cabeza para ver cómo responde la balanza.")
    print("Presiona 'C' para fijar un nuevo centro | 'Q' para salir.\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - start_time) * 1000)

            result = landmarker.detect_for_video(mp_image, ts_ms)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                # Puntos de la cabeza
                nose = get_p2(landmarks[NOSE_TIP], w, h)
                forehead = get_p2(landmarks[FOREHEAD], w, h)

                # Si no hay anclaje previo o el usuario presiona 'C', se guarda el centro
                if anchor_center is None:
                    anchor_center = nose.copy()

                # 1. CALCULAR DESPLAZAMIENTO respecto al punto de origen
                offset_x = nose[0] - anchor_center[0]
                offset_y = nose[1] - anchor_center[1]
                distance = int(np.sqrt(offset_x**2 + offset_y**2))

                # Color según el margen
                is_centered = distance <= max_allowed_drift
                status_color = (0, 255, 0) if is_centered else (0, 0, 255)

                # 2. DIBUJAR MARGEN SEGURO (Cuadro centrado en el punto ancla)
                cv2.rectangle(
                    frame,
                    (anchor_center[0] - max_allowed_drift, anchor_center[1] - max_allowed_drift),
                    (anchor_center[0] + max_allowed_drift, anchor_center[1] + max_allowed_drift),
                    status_color,
                    2,
                )

                # 3. DIBUJAR "BALANZA" SOBRE LA CABEZA (40px arriba de la frente)
                bar_center_x = forehead[0]
                bar_center_y = max(30, forehead[1] - 40)
                bar_width = 120

                # Base fija de la balanza
                cv2.line(
                    frame,
                    (bar_center_x - bar_width // 2, bar_center_y),
                    (bar_center_x + bar_width // 2, bar_center_y),
                    (200, 200, 200),
                    2,
                )
                cv2.line(
                    frame,
                    (bar_center_x, bar_center_y - 8),
                    (bar_center_x, bar_center_y + 8),
                    (200, 200, 200),
                    2,
                )

                # Indicador móvil de la balanza
                indicator_x = int(bar_center_x + (offset_x * 0.8))
                indicator_x = np.clip(
                    indicator_x,
                    bar_center_x - bar_width // 2,
                    bar_center_x + bar_width // 2,
                )

                # Pelota/Nivel de la balanza
                cv2.circle(frame, (indicator_x, bar_center_y), 8, status_color, -1)
                cv2.circle(frame, (indicator_x, bar_center_y), 10, (255, 255, 255), 1)

                # Estado
                status_text = "CABEZA CENTRADA" if is_centered else "¡FUERA DE MARGEN!"
                cv2.putText(frame, status_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(
                    frame,
                    f"Desviacion: {distance}px / Max: {max_allowed_drift}px",
                    (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (220, 220, 220),
                    1,
                )

            cv2.imshow("Paso 1: Balanza de Cabeza", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("c"), ord("C")):
                anchor_center = None  # Reestablece el centro de la cabeza

            if key in (27, ord("q")):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()