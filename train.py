import os
print("Connexion à Hugging Face réussie.")
token = os.getenv("HF_TOKEN")
if token:
    print("Pipeline prêt : HF_TOKEN détecté.")
else:
    print("Erreur : HF_TOKEN manquant.")
