import os
import subprocess
from huggingface_hub import snapshot_download

def convert_to_gguf():
    # 1. Définir le chemin absolu
    base_dir = os.getcwd()
    model_dir = os.path.join(base_dir, "fine_tuned_model")
    
    # 2. S'assurer que le répertoire existe
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Téléchargement du modèle dans : {model_dir}")
    snapshot_download(repo_id="ouzaif/Expert-Cyber-Droit-FineTuned", local_dir=model_dir, token=os.getenv("HF_TOKEN"))
    
    # 3. Vérification de sécurité
    if not os.listdir(model_dir):
        raise Exception("Le répertoire du modèle est vide après le téléchargement !")
    
    print("Clonage de llama.cpp...")
    if not os.path.exists("llama.cpp"):
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"], check=True)
    
    print("Conversion...")
    output_gguf = os.path.join(base_dir, "Expert-Cyber-Droit.gguf")
    convert_script = os.path.join(base_dir, "llama.cpp", "convert_hf_to_gguf.py")
    
    convert_cmd = [
        "python3", convert_script,
        model_dir,
        "--outfile", output_gguf,
        "--outtype", "f16",
        "--model-type", "qwen2"
    ]
    subprocess.run(convert_cmd, check=True)
    
    print("Upload...")
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=output_gguf,
        path_in_repo="Expert-Cyber-Droit.gguf",
        repo_id="ouzaif/Expert-Cyber-Droit-GGUF",
        token=os.getenv("HF_TOKEN")
    )

if __name__ == "__main__":
    convert_to_gguf()
