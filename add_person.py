import os
import pickle
from deepface import DeepFace
from deepface import DeepFace

# NOME DA NOVA PESSOA
person_name = "NOME_DA_PESSOA_AQUI"

# Caminho para a pasta dessa pessoa
person_path = os.path.join("Photos", person_name)

# Carrega o banco existente
with open("embeddings_database.pkl", "rb") as f:
    database = pickle.load(f)

model = Facenet.loadModel()

for img_name in os.listdir(person_path):
    img_path = os.path.join(person_path, img_name)

    try:
        emb = DeepFace.represent(
            img_path=img_path,
            model=model,
            enforce_detection=True
        )[0]["embedding"]

        database.append({
            "embedding": emb,
            "label": person_name
        })

        print(f"Embedding adicionado: {person_name} - {img_name}")

    except Exception as e:
        print("Erro:", e)

# Salva de volta o banco
with open("embeddings_database.pkl", "wb") as f:
    pickle.dump(database, f)

print("Pessoa adicionada ao banco com sucesso!")
