import cv2
import pickle
import numpy as np
from deepface import DeepFace

# Carrega o banco treinado
with open("embeddings_database.pkl", "rb") as f:
    database = pickle.load(f)

model = "Facenet"
threshold = 0.4  # recomendado para Facenet

# Inicializa a câmera com DirectShow para Windows
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Carregando Haar Cascade para detecção de rosto
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def distance(v1, v2):
    return np.linalg.norm(v1 - v2)

print("Reconhecimento iniciado...")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Erro ao capturar frame da câmera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    nome = "Unknown"

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Redimensiona o rosto para tamanho mínimo do modelo
        face_resized = cv2.resize(face, (160, 160))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

        try:
            emb = DeepFace.represent(
                img_path=face_rgb,
                model_name=model,
                enforce_detection=False
            )[0]["embedding"]

            smallest_dist = float("inf")
            predicted_label = "Unknown"

            for item in database:
                dist = distance(np.array(emb), np.array(item["embedding"]))
                if dist < smallest_dist:
                    smallest_dist = dist
                    predicted_label = item["label"]

            if smallest_dist < threshold:
                nome = predicted_label

        except Exception as e:
            nome = "Unknown"
            # opcional: print("Erro ao gerar embedding:", e)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(frame, nome, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Reconhecimento Facial", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
