#import necessary libraries
import numpy as np
import cv2
import requests
from math import cos, sin
import mediapipe as mp

# Mediapipe FaceMesh instance
mp_face_mesh = mp.solutions.face_mesh # type: ignore

# Map angles to [-π, π] )
def map_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Landmark Preprocessing
def preprocess(face, width=450, height=450):
    """
    Convert Mediapipe landmarks into a normalized array
    - Centers landmarks around the nose
    - Normalizes X and Y coordinates
    """
    x_val = [lm.x * width for lm in face.landmark]
    y_val = [lm.y * height for lm in face.landmark]
    # Center around nose landmark (id = 1)
    x_val = np.array(x_val) - np.mean(x_val[1])
    y_val = np.array(y_val) - np.mean(y_val[1])
    # Normalize
    x_val = x_val / x_val.max() if x_val.max() != 0 else x_val
    y_val = y_val / y_val.max() if y_val.max() != 0 else y_val
    return np.concatenate([x_val, y_val])


#Axis Drawing (3D Pose Visualization)
def draw_axis(img, pitch, yaw, roll, tdx=None, tdy=None, size=100,thickness = 10):
    """
    Draw pitch, yaw, roll axes on an image.
    Anchored at the nose tip (or image center if not provided).
    """
    yaw = -yaw # invert yaw for drawing

    if tdx is None or tdy is None:
        height, width = img.shape[:2]
        tdx, tdy = width // 2, height // 2  # fallback

    # compute axis endpoints
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    # draw lines
    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), thickness)  # X axis
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), thickness)  # Y axis
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), thickness)  # Z axis
    return img

# Image Processing
def process_image(img_path_or_url, model, output_path="outputs/api_result.jpg"):
    """Process a single image (local path or URL) and save result."""
    if img_path_or_url.startswith("http"):
        response = requests.get(img_path_or_url)
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(img_array, -1)
    else:
        image = cv2.imread(img_path_or_url)

    if image is None:
        raise ValueError("❌ Unable to load image")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh( # type: ignore
        static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        # Preprocess landmarks and predict pose
        marks = preprocess(face, rgb_image.shape[1], rgb_image.shape[0])
        pred_angles = model.predict(marks.reshape(1, -1))[0]
        pitch, yaw, roll = pred_angles

        # Preprocess landmarks and predict pose
        nose = face.landmark[1]
        h, w, _ = image.shape
        tdx, tdy = int(nose.x * w), int(nose.y * h)

        # Draw pose axes
        img_with_pred = draw_axis(image.copy(), pitch, yaw, roll, tdx=tdx, tdy=tdy)
        cv2.imwrite(output_path, img_with_pred)

        return output_path, (float(pitch), float(yaw), float(roll))
    else:
        raise ValueError("❌ No face detected in image")


def process_video(video_path_or_url, model, output_path="outputs/api_result.mp4"):
    """Process a video (local path or URL) and save result."""
    if video_path_or_url.startswith("http"):
        response = requests.get(video_path_or_url)
        with open("temp_video.mp4", "wb") as f:
            f.write(response.content)
        cap = cv2.VideoCapture("temp_video.mp4")
    else:
        cap = cv2.VideoCapture(video_path_or_url)

    if not cap.isOpened():
        raise ValueError("❌ Unable to load video")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    with mp_face_mesh.FaceMesh( # type: ignore
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                 # Preprocess + predict
                marks = preprocess(face, frame.shape[1], frame.shape[0])
                pred_angles = model.predict(marks.reshape(1, -1))[0]
                pitch, yaw, roll = pred_angles
                # Nose tip anchor
                nose = face.landmark[1]
                h, w, _ = frame.shape
                tdx, tdy = int(nose.x * w), int(nose.y * h)
                # Draw axes
                frame_with_axes = draw_axis(frame.copy(), pitch, yaw, roll, tdx=tdx, tdy=tdy,size=80,thickness=5)
                out.write(frame_with_axes)

    cap.release()
    out.release()
    return output_path
