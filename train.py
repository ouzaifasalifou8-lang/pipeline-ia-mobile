import os
import subprocess

def run_training():
    print("Démarrage de l'entraînement via la ligne de commande...")
    
    # Commande autotrain pour le fine-tuning LLM
    cmd = [
        "autotrain", "llm",
        "--train",
        "--model", "Qwen/Qwen2-1.5B",
        "--data-path", "ouzaif/Expert-Cyber-Droit-Dataset",
        "--lr", "2e-5",
        "--epochs", "3",
        "--batch-size", "2",
        "--block-size", "512",
        "--trainer", "sft",
        "--target-modules", "all-linear",
        "--push-to-hub",
        "--repo-id", "ouzaif/Expert-Cyber-Droit-FineTuned",
        "--token", os.getenv("HF_TOKEN")
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        exit(1)

if __name__ == "__main__":
    run_training()
