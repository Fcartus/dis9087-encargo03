"""
Sistema Definitivo: Control de Iris con Calibración Multivariable
(Ojos, Nariz, Labios y Postura 3D de Cabeza), CLAHE, Balanza y HUD Central.
"""

from pathlib import Path
import atexit
import random
import shutil
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

# --- GESTIÓN Y LIMPIEZA AUTOMÁTICA DE CACHÉ ---
ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "temp_cache"
CACHE_DIR.mkdir(exist_ok=True)


def cleanup_cache():
    if CACHE_DIR.exists():
        try:
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            print("\n[INFO] Carpeta 'temp_cache' y residuos eliminados.")
        except Exception as e:
            print(f"\n[AVISO] Error al eliminar caché: {e}")


atexit.register(cleanup_cache)

# --- CARGA DEL MODELO DESDE MEMORIA ---
MODELS_DIR = ROOT / "models"
model_file = None
if MODELS_DIR.exists():
    for f in MODELS_DIR.iterdir():
        if "face_landmarker" in f.name.lower():
            model_file = f
            break

if not model_file or not model_file.exists():
    print("\n[ERROR] No se encontró el archivo del modelo en 'models'.")
    input("Presiona ENTER para salir...")
    exit()

model_bytes = model_file.read_bytes()

# --- ÍNDICES DE LANDMARKS CLAVE ---
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
LEFT_FACE = 234
RIGHT_FACE = 454

MOUTH_LEFT = 61
MOUTH_RIGHT = 291

LEFT_IRIS = 468
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

RIGHT_IRIS = 473
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374


def enhance_contrast_low_light(frame):
    """Ecualización adaptativa de contraste (CLAHE) para baja iluminación."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def get_p2(lm, width, height):
    return np.array([int(lm.x * width), int(lm.y * height)])


def extract_multivariable_features(landmarks, w, h):
    """
    Extrae un vector de 12 características combinando:
    1. Ojos (Ratios del iris)
    2. Nariz (Posición absoluta)
    3. Labios (Inclinación / Roll)
    4. Cabeza (Yaw y Pitch)
    """
    # 1. OJOS (Ratio Iris)
    l_iris = get_p2(landmarks[LEFT_IRIS], w, h)
    l_left = get_p2(landmarks[LEFT_EYE_LEFT], w, h)
    l_right = get_p2(landmarks[LEFT_EYE_RIGHT], w, h)
    l_top = get_p2(landmarks[LEFT_EYE_TOP], w, h)
    l_bottom = get_p2(landmarks[LEFT_EYE_BOTTOM], w, h)

    l_h_ratio = (l_iris[0] - l_left[0]) / (l_right[0] - l_left[0] + 1e-6)
    l_v_ratio = (l_iris[1] - l_top[1]) / (l_bottom[1] - l_top[1] + 1e-6)

    r_iris = get_p2(landmarks[RIGHT_IRIS], w, h)
    r_left = get_p2(landmarks[RIGHT_EYE_LEFT], w, h)
    r_right = get_p2(landmarks[RIGHT_EYE_RIGHT], w, h)
    r_top = get_p2(landmarks[RIGHT_EYE_TOP], w, h)
    r_bottom = get_p2(landmarks[RIGHT_EYE_BOTTOM], w, h)

    r_h_ratio = (r_iris[0] - r_left[0]) / (r_right[0] - r_left[0] + 1e-6)
    r_v_ratio = (r_iris[1] - r_top[1]) / (r_bottom[1] - r_top[1] + 1e-6)

    h_eye = (l_h_ratio + r_h_ratio) / 2.0
    v_eye = (l_v_ratio + r_v_ratio) / 2.0

    # 2. NARIZ
    nose = landmarks[NOSE_TIP]
    nose_x, nose_y = nose.x, nose.y

    # 3. CABEZA (Yaw y Pitch)
    f_left = landmarks[LEFT_FACE]
    f_right = landmarks[RIGHT_FACE]
    forehead = landmarks[FOREHEAD]
    chin = landmarks[CHIN]

    face_w = abs(f_right.x - f_left.x) + 1e-6
    face_h = abs(chin.y - forehead.y) + 1e-6

    yaw = (nose.x - f_left.x) / face_w
    pitch = (nose.y - forehead.y) / face_h

    # 4. LABIOS (Roll / Inclinación)
    m_left = landmarks[MOUTH_LEFT]
    m_right = landmarks[MOUTH_RIGHT]
    roll = np.arctan2(m_right.y - m_left.y, m_right.x - m_left.x)

    # VECTOR DE CARACTERÍSTICAS
    features = np.array([
        1.0,
        h_eye,
        v_eye,
        yaw,
        pitch,
        roll,
        nose_x,
        nose_y,
        h_eye * yaw,  # Compensación horizontal
        v_eye * pitch,  # Compensación vertical
        h_eye**2,
        v_eye**2,
    ])

    return features


def filter_outliers(samples):
    """Limpia muestras fuera de rango generadas por parpadeos o saltos."""
    if len(samples) < 10:
        return samples

    features_matrix = np.array([s[0] for s in samples])
    means = np.mean(features_matrix, axis=0)
    stds = np.std(features_matrix, axis=0) + 1e-6

    valid_mask = np.all(
        np.abs(features_matrix - means) <= 1.5 * stds, axis=1
    )
    filtered = [samples[i] for i in range(len(samples)) if valid_mask[i]]

    return filtered if len(filtered) > 5 else samples


def train_multivariable_regression(calibration_samples):
    """Entrena la matriz de mínimos cuadrados sobre el vector multivariable."""
    A = np.array([s[0] for s in calibration_samples])
    X = np.array([s[1] for s in calibration_samples])
    Y = np.array([s[2] for s in calibration_samples])

    coeff_x, _, _, _ = np.linalg.lstsq(A, X, rcond=None)
    coeff_y, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)

    return coeff_x, coeff_y


def predict_screen_position(features, coeff_x, coeff_y):
    pred_x = np.dot(features, coeff_x)
    pred_y = np.dot(features, coeff_y)
    return pred_x, pred_y


def draw_hud_crosshair(canvas, width, height, alpha=0.65):
    """HUD central interlineado con opacidad al 65%."""
    overlay = canvas.copy()
    cx, cy = width // 2, height // 2
    color = (230, 230, 230)
    dash_len, gap_len = 12, 8

    for x in range(0, width, dash_len + gap_len):
        cv2.line(overlay, (x, cy), (min(x + dash_len, width), cy), color, 1)

    for y in range(0, height, dash_len + gap_len):
        cv2.line(overlay, (cx, y), (cx, min(y + dash_len, height)), color, 1)

    cv2.circle(overlay, (cx, cy), 14, color, 1)
    cv2.circle(overlay, (cx, cy), 3, (0, 255, 255), -1)

    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)


def main():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_bytes),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
    )

    landmarker = FaceLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)

    window_name = "Sistema de Iris Multivariable - Calibración 3D Pose + Iris"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )

    start_time = time.time()

    anchor_center = None
    max_allowed_drift = 45

    steps = [0.125, 0.375, 0.625, 0.875]
    calibration_targets = []
    idx = 1
    for row in steps:
        for col in steps:
            calibration_targets.append(
                {"name": f"{idx}/16", "px": col, "py": row}
            )
            idx += 1

    calib_index = 0
    all_calibration_samples = []

    is_sampling = False
    sample_start_time = 0
    current_point_samples = []
    sampling_duration = 2.2

    state = "CALIBRATION"
    coeff_x, coeff_y = None, None
    smooth_x, smooth_y = None, None
    alpha = 0.15

    target_zone = random.randint(0, 15)
    gaze_dwell_start = None
    score = 0
    dwell_duration = 1.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = enhance_contrast_low_light(frame)
            h_img, w_img, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - start_time) * 1000)

            result = landmarker.detect_for_video(mp_image, ts_ms)

            features = None
            is_centered = False
            offset_x = 0

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                features = extract_multivariable_features(
                    landmarks, w_img, h_img
                )

                nose = get_p2(landmarks[NOSE_TIP], w_img, h_img)
                forehead = get_p2(landmarks[FOREHEAD], w_img, h_img)

                if anchor_center is None:
                    anchor_center = nose.copy()

                offset_x = nose[0] - anchor_center[0]
                offset_y = nose[1] - anchor_center[1]
                distance = int(np.sqrt(offset_x**2 + offset_y**2))
                is_centered = distance <= max_allowed_drift

            canvas = frame.copy()

            # 1. DIBUJAR HUD CENTRAL
            draw_hud_crosshair(canvas, w_img, h_img, alpha=0.65)

            # 2. DIBUJAR BALANZA
            if result.face_landmarks:
                status_color = (0, 255, 0) if is_centered else (0, 0, 255)

                cv2.rectangle(
                    canvas,
                    (
                        anchor_center[0] - max_allowed_drift,
                        anchor_center[1] - max_allowed_drift,
                    ),
                    (
                        anchor_center[0] + max_allowed_drift,
                        anchor_center[1] + max_allowed_drift,
                    ),
                    status_color,
                    2,
                )

                bar_cx = forehead[0]
                bar_cy = max(30, forehead[1] - 35)
                bar_w = 120

                cv2.line(
                    canvas,
                    (bar_cx - bar_w // 2, bar_cy),
                    (bar_cx + bar_w // 2, bar_cy),
                    (200, 200, 200),
                    2,
                )
                indicator_x = int(
                    np.clip(
                        bar_cx + (offset_x * 0.8),
                        bar_cx - bar_w // 2,
                        bar_cx + bar_w // 2,
                    )
                )

                cv2.circle(
                    canvas, (indicator_x, bar_cy), 7, status_color, -1
                )

            # 3. ESTADO: CALIBRACIÓN EXTENSA MULTIVARIABLE
            if state == "CALIBRATION":
                target = calibration_targets[calib_index]
                tx, ty = int(target["px"] * w_img), int(target["py"] * h_img)

                if is_sampling:
                    if not is_centered:
                        cv2.putText(
                            canvas,
                            "¡MANTÉN LA CABEZA CENTRADA EN LA BALANZA!",
                            (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                        sample_start_time = time.time() - (
                            len(current_point_samples)
                            * (sampling_duration / 60)
                        )
                    else:
                        elapsed = time.time() - sample_start_time
                        progress = min(1.0, elapsed / sampling_duration)

                        if features is not None:
                            current_point_samples.append((features, tx, ty))

                        r_anim = int(35 * (1 - progress * 0.5))
                        cv2.circle(canvas, (tx, ty), r_anim, (0, 255, 0), -1)
                        cv2.circle(canvas, (tx, ty), 40, (255, 255, 255), 3)

                        cv2.putText(
                            canvas,
                            f"CALIBRANDO MULTIVARIABLE {target['name']}... {int(progress * 100)}%",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )

                        if elapsed >= sampling_duration:
                            is_sampling = False
                            cleaned_samples = filter_outliers(
                                current_point_samples
                            )
                            all_calibration_samples.extend(cleaned_samples)
                            current_point_samples = []
                            calib_index += 1

                            if calib_index >= len(calibration_targets):
                                coeff_x, coeff_y = (
                                    train_multivariable_regression(
                                        all_calibration_samples
                                    )
                                )
                                state = "GAME"
                                print(
                                    f"\n[OK] Modelo Multivariable Entrenado ({len(all_calibration_samples)} muestras)."
                                )

                else:
                    cv2.circle(canvas, (tx, ty), 22, (0, 0, 255), -1)
                    cv2.circle(canvas, (tx, ty), 26, (255, 255, 255), 2)

                    cv2.putText(
                        canvas,
                        f"Punto Multivariable {target['name']}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        canvas,
                        "Mira el punto fijo y presiona 'ESPACIO' ('C' para recentrar)",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (250, 250, 250),
                        1,
                    )

            # 4. ESTADO: JUEGO GRILLA 4x4
            elif state == "GAME":
                for i in range(1, 4):
                    cv2.line(
                        canvas,
                        (int(i * w_img / 4), 0),
                        (int(i * w_img / 4), h_img),
                        (70, 70, 70),
                        1,
                    )
                    cv2.line(
                        canvas,
                        (0, int(i * h_img / 4)),
                        (w_img, int(i * h_img / 4)),
                        (70, 70, 70),
                        1,
                    )

                active_col, active_row = 0, 0
                if features is not None:
                    raw_x, raw_y = predict_screen_position(
                        features, coeff_x, coeff_y
                    )

                    if smooth_x is None:
                        smooth_x, smooth_y = raw_x, raw_y
                    else:
                        smooth_x = alpha * raw_x + (1 - alpha) * smooth_x
                        smooth_y = alpha * raw_y + (1 - alpha) * smooth_y

                    clamp_x = np.clip(smooth_x, 0, w_img - 1)
                    clamp_y = np.clip(smooth_y, 0, h_img - 1)

                    active_col = int(np.clip(clamp_x // (w_img / 4), 0, 3))
                    active_row = int(np.clip(clamp_y // (h_img / 4), 0, 3))

                    cv2.circle(
                        canvas,
                        (int(clamp_x), int(clamp_y)),
                        10,
                        (0, 255, 0),
                        -1,
                    )
                    cv2.circle(
                        canvas,
                        (int(clamp_x), int(clamp_y)),
                        14,
                        (255, 255, 255),
                        2,
                    )

                current_zone_index = active_row * 4 + active_col

                t_col, t_row = target_zone % 4, target_zone // 4
                tx1, ty1 = int(t_col * w_img / 4), int(t_row * h_img / 4)
                tx2, ty2 = int((t_col + 1) * w_img / 4), int(
                    (t_row + 1) * h_img / 4
                )

                overlay = canvas.copy()
                cv2.rectangle(
                    overlay, (tx1, ty1), (tx2, ty2), (0, 255, 255), -1
                )
                cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
                cv2.putText(
                    canvas,
                    f"OBJETIVO #{target_zone + 1}",
                    (tx1 + 10, ty1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                if current_zone_index == target_zone and is_centered:
                    if gaze_dwell_start is None:
                        gaze_dwell_start = time.time()

                    elapsed = time.time() - gaze_dwell_start
                    progress = min(1.0, elapsed / dwell_duration)

                    bar_w = int((tx2 - tx1) * progress)
                    cv2.rectangle(
                        canvas,
                        (tx1, ty2 - 12),
                        (tx1 + bar_w, ty2),
                        (0, 255, 0),
                        -1,
                    )

                    if elapsed >= dwell_duration:
                        score += 1
                        gaze_dwell_start = None
                        target_zone = (target_zone + random.randint(1, 15)) % 16
                else:
                    gaze_dwell_start = None

                cv2.putText(
                    canvas,
                    f"PUNTOS: {score}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

                if not is_centered:
                    cv2.putText(
                        canvas,
                        "¡MANTÉN LA CABEZA CENTRADA EN LA BALANZA!",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("c"), ord("C")):
                anchor_center = None

            if (
                key == 32
                and state == "CALIBRATION"
                and not is_sampling
                and features is not None
            ):
                if is_centered:
                    is_sampling = True
                    sample_start_time = time.time()
                    current_point_samples = []

            if key in (ord("r"), ord("R")):
                state = "CALIBRATION"
                calib_index = 0
                all_calibration_samples = []
                is_sampling = False
                score = 0

            if key in (27, ord("q")):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()