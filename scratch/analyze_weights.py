import torch
from transformers import AutoModelForCausalLM
import numpy as np

def analyze():
    print("Carregando modelo para análise de concentração...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-e4b-it", 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    layers = model.model.language_model.layers
    
    print(f"{'Layer':>6} | {'Mean':>8} | {'Std':>8} | {'Max/Mean':>8} | {'Gini':>8}")
    print("-" * 50)
    
    for i, layer in enumerate(layers):
        weights = layer.mlp.gate_proj.weight.data.float().norm(dim=1).cpu().numpy()
        
        mean = np.mean(weights)
        std = np.std(weights)
        max_val = np.max(weights)
        
        # Gini Coefficient (0 = igualdade total, 1 = concentração total em 1 neurônio)
        sorted_weights = np.sort(weights)
        n = len(weights)
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * sorted_weights)) / (n * np.sum(sorted_weights))
        
        print(f"{i:02d}    | {mean:8.4f} | {std:8.4f} | {max_val/mean:8.2f} | {gini:8.4f}")

if __name__ == "__main__":
    analyze()
