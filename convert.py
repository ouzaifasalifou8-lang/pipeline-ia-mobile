import os
import subprocess
from huggingface_hub import snapshot_download

def convert_to_gguf():
    print("Étape 1 : Téléchargement du modèle fine-tuné depuis Hugging Face...")
    model_id = "ouzaif/Expert-Cyber-Droit-FineTuned"
    token = os.getenv("HF_TOKEN")
    
    # Téléchargement local dans un dossier 'model_dir'
    model_dir = "./fine_tuned_model"
    snapshot_download(repo_id=model_id, local_dir=model_dir, token=token)
    
    print("Étape 2 : Clonant llama.cpp pour la conversion...")
    subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"], check=True)
    
    print("Étape 3 : Installation des dépendances Python de llama.cpp...")
    subprocess.run(["pip", "install", "-r", "llama.cpp/requirements.txt"], check=True)
    
    print("Étape 4 : Lcette conversion du modèle au format GGUF (FP16)...")
    output_gguf = "./Expert-Cyber-Droit.gguf"
    
    # Utilisation du script de conversion de llama.cpp
    convert_cmd = [
        "python", "llama.cpp/convert_hf_to_gguf.py",
        model_dir,
        "--outfile", output_gguf,
        "--outtype", "f16"
    ]
    subprocess.run(convert_cmd, check=True)
    
    print("Étape 5 : Pousser le fichier GGUF converti vers un nouveau dépôt...")
    # On crée/utilise un dépôt dédié aux GGUF pour votre application mobile
    target_repo = "ouzaif/Expert-Cyber-Droit-GGUF"
    
    from huggingface_hub import HfApi
    api = HfApi()
    
    # Création du dépôt distant s'il n'existe pas
    try:
        api.create_repo(repo_id=target_repo, token=token, private=False)
    except Exception:
        print("Le dépôt GGUF existe déjà, poursuite...")
        
    api.upload_file(
        path_or_fileobj=output_gguf,
        path_in_repo="Expert-Cyber-Droit.gguf",
        repo_id=target_repo,
        token=token
    )
    print("Modèle converti et poussé avec succès sur ouzaif/Expert-Cyber-Droit-GGUF !")

if __name__ == "__main__":
    convert_to_gguf()
