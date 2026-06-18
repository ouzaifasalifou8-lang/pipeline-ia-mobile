import os
from autotrain.trainers.text_classification import LLMTrainer, LLMParams

def run_autotrain():
    print("Vérification du token Hugging Face...")
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("ERREUR : Le token HF_TOKEN n'est pas détecté dans l'environnement.")
        return
        
    print("Configuration des paramètres d'entraînement...")
    
    # Paramètres du fine-tuning (modèle, dataset, etc.)
    params = LLMParams(
        train_data="ouzaif/Expert-Cyber-Droit-Dataset",
        model="Qwen/Qwen2-1.5B",
        lr=2e-5,
        epochs=3,
        batch_size=2,
        block_size=512,
        trainer="sft",
        target_modules="all-linear",
        push_to_hub=True,
        repo_id="ouzaif/Expert-Cyber-Droit-FineTuned",
        token=hf_token
    )
    
    trainer = LLMTrainer(params)
    
    print("Lancement de l'entraînement sur les serveurs GPU distants...")
    trainer.train()
    print("Entraînement terminé et modèle poussé sur le Hub avec succès !")

if __name__ == "__main__":
    run_autotrain()
