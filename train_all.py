import os
import cv2
import pickle
from deepface import DeepFace
import numpy as np

# Caminho da pasta com as fotos originais
dataset_path = r"C:\Users\Gi\OneDrive\Voitureproject\FaceRecognition\Photos"

# Caminho para salvar imagens processadas
processed_path = r"C:\Users\Gi\OneDrive\Voitureproject\FaceRecognition\Processed"
os.makedirs(processed_path, exist_ok=True)

# Inicializa Haar Cascade para detecção de rosto
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Lista para armazenar embeddings
embeddings_database = []

# Itera sobre cada pasta de pessoa
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    processed_person_folder = os.path.join(processed_path, person_name)
    os.makedirs(processed_person_folder, exist_ok=True)

    for file_name in os.listdir(person_folder):
        file_path = os.path.join(person_folder, file_name)
        try:
            img = cv2.imread(file_path)
            if img is None:
                print(f"Erro ao ler {file_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                print(f"Rosto não detectado em {file_path}")
                continue

            # Pega o maior rosto detectado
            x, y, w, h = max(faces, key=lambda bbox: bbox[2]*bbox[3])
            face = img[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (160, 160))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            # Salva o rosto processado
            processed_file_path = os.path.join(processed_person_folder, file_name)
            cv2.imwrite(processed_file_path, cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))

            # Gera embedding
            emb = DeepFace.represent(
                img_path=face_rgb,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]

            embeddings_database.append({
                "label": person_name,
                "embedding": emb
            })

            print(f"Embedding gerado para {person_name} - {file_name}")

        except Exception as e:
            print(f"Erro ao processar {file_path}: {e}")

# Salva o banco de embeddings
with open("embeddings_database.pkl", "wb") as f:
    pickle.dump(embeddings_database, f)

print("Banco de embeddings gerado com sucesso!")
