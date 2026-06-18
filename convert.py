import os
import subprocess
from huggingface_hub import snapshot_download

def convert_to_gguf():
    print("Étape 1 : Téléchargement du modèle...")
    model_id = "ouzaif/Expert-Cyber-Droit-FineTuned"
    model_dir = "fine_tuned_model"
    snapshot_download(repo_id=model_id, local_dir=model_dir, token=os.getenv("HF_TOKEN"))
    
    print("Étape 2 : Clonage de llama.cpp...")
    if not os.path.exists("llama.cpp"):
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"], check=True)
    
    print("Étape 3 : Conversion directe avec python...")
    output_gguf = "Expert-Cyber-Droit.gguf"
    
    # Utilisation de la conversion directe sans dépendre autant de config.json
    convert_cmd = [
        "python3", "llama.cpp/convert_hf_to_gguf.py",
        model_dir,
        "--outfile", output_gguf,
        "--outtype", "f16"
    ]
    subprocess.run(convert_cmd, check=True)
    
    print("Étape 4 : Upload vers Hugging Face...")
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
