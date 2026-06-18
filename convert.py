import os
import subprocess
from huggingface_hub import snapshot_download

def convert_to_gguf():
    base_dir = os.getcwd()
    model_dir = os.path.join(base_dir, "fine_tuned_model")
    
    # Téléchargement
    snapshot_download(repo_id="ouzaif/Expert-Cyber-Droit-FineTuned", local_dir=model_dir, token=os.getenv("HF_TOKEN"))
    
    # Conversion (sans l'argument --model-type qui cause l'erreur)
    output_gguf = os.path.join(base_dir, "Expert-Cyber-Droit.gguf")
    convert_script = os.path.join(base_dir, "llama.cpp", "convert_hf_to_gguf.py")
    
    # On laisse le script détecter le modèle tout seul
    convert_cmd = [
        "python3", convert_script,
        model_dir,
        "--outfile", output_gguf,
        "--outtype", "f16"
    ]
    subprocess.run(convert_cmd, check=True)
    
    # Upload
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
