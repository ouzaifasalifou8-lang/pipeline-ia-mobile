import os
import subprocess
from huggingface_hub import snapshot_download

def convert_to_gguf():
    model_id = "ouzaif/Expert-Cyber-Droit-FineTuned"
    model_dir = "fine_tuned_model"
    
    print("Téléchargement du modèle...")
    snapshot_download(repo_id=model_id, local_dir=model_dir, token=os.getenv("HF_TOKEN"))
    
    print("Conversion forcée en Qwen2...")
    output_gguf = "Expert-Cyber-Droit.gguf"
    
    # On ajoute --model-type qwen2 pour lever l'ambiguïté
    convert_cmd = [
        "python3", "llama.cpp/convert_hf_to_gguf.py",
        model_dir,
        "--outfile", output_gguf,
        "--outtype", "f16",
        "--model-type", "qwen2"
    ]
    
    subprocess.run(convert_cmd, check=True)
    
    print("Upload vers le hub...")
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
