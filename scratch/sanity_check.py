import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from unified.mock_gemma4_e4b import MockGemma4E4B
from deepslice_moe_converter import DeepSliceConverter

def main():
    print("Iniciando Verificação de Sanidade (Norma do Output)...")
    
    # Usar Mock Model para verificação rápida sem CUDA
    model = MockGemma4E4B(lite=True)
    device = "cpu"
    
    # Converter para MoE
    converter = DeepSliceConverter(
        model=model,
        shared_ratio=0.5,
        num_routed_experts=8,
        num_experts_per_tok=2,
        use_relu2=False
    )
    model = converter.convert()
    
    # Input dummy (usar hidden_size real do mock)
    hidden_size = model._dims["hidden"]
    x = torch.randn(1, 4, hidden_size, dtype=torch.float32)
    
    # Acessar a primeira layer (que agora tem MoE)
    # No MockGemma4E4B, as layers estão acessíveis via text_layers property
    layer_moe = model.text_layers[0]
        
    with torch.no_grad():
        out = layer_moe.mlp(x)
        norm = out.norm().item()
        print(f"MoE output norm: {norm:.2f}")
        
        if norm < 500:
            print("✅ SANITY CHECK PASSED: Norma dentro da faixa esperada.")
        else:
            print(f"❌ SANITY CHECK FAILED: Norma muito alta ({norm:.2f}). Overflow detectado.")

if __name__ == "__main__":
    main()
