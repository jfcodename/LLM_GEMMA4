import torch
import torch.nn as nn
import logging
from typing import List, Dict

from unified.modules import DeepSliceMoE, PrunedMLP, Mamba2Router
from structural_pruning import NeuronImportanceCalibrator

logger = logging.getLogger(__name__)

class DeepSliceConverter:
    """
    Motor de MoEification: converte a MLP densa monolítica em um DeepSliceMoE.
    Faz a triagem dos neurônios baseada em importância, alocando o top-%
    para o Shared Expert e fragmentando o resto em Routed Experts.
    """
    def __init__(
        self, 
        model: nn.Module, 
        shared_ratio: float = 0.25, 
        num_routed_experts: int = 8,
        num_experts_per_tok: int = 2,
        use_relu2: bool = False
    ):
        self.model = model
        self.shared_ratio = shared_ratio
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.use_relu2 = use_relu2
        self.device = next(model.parameters()).device
        
    def convert(self):
        calibrator = NeuronImportanceCalibrator(self.model, device=self.device)
        logger.info("Calibrando neurônios via Proxy (Gate Norm) para alocação MoE...")
        importance_scores = calibrator.calibrate_with_gate(keep_ratio=1.0)
        
        # Encontra todas as camadas de atenção do Gemma/Mock
        layers = []
        for name, module in self.model.named_modules():
            # Gemma 4 structure (DecoderLayer) has an 'mlp' attribute
            if hasattr(module, 'mlp') and hasattr(module.mlp, 'gate_proj'):
                layers.append(module)
                
        for layer_idx, layer in enumerate(layers):
            if layer_idx not in importance_scores:
                continue
                
            scores = importance_scores[layer_idx]
            n_total = scores.size(0)
            
            # Ordena do mais importante para o menos importante
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            
            n_shared = int(n_total * self.shared_ratio)
            n_routed_total = n_total - n_shared
            n_per_expert = n_routed_total // self.num_routed_experts
            
            shared_idx = sorted_indices[:n_shared]
            routed_indices = sorted_indices[n_shared:]
            
            def create_expert(indices):
                # Mantém a ordem original dos canais de ativação
                indices = indices.sort()[0] 
                gate_w = layer.mlp.gate_proj.weight.data[indices].clone()
                up_w   = layer.mlp.up_proj.weight.data[indices].clone()
                down_w = layer.mlp.down_proj.weight.data[:, indices].clone()
                hidden_size = layer.mlp.gate_proj.in_features
                
                return PrunedMLP(
                    hidden_size=hidden_size,
                    kept_neurons=len(indices),
                    gate_proj_weight=gate_w,
                    up_proj_weight=up_w,
                    down_proj_weight=down_w,
                    use_relu2=self.use_relu2,
                    neuron_indices=indices.cpu(),
                )

            # 1. Cria Especialista Fixo (Shared)
            shared_expert = create_expert(shared_idx)
            
            # 2. Cria Especialistas Roteados (Experts)
            routed_experts_list = []
            for i in range(self.num_routed_experts):
                start = i * n_per_expert
                end = start + n_per_expert if i < self.num_routed_experts - 1 else len(routed_indices)
                expert_idx = routed_indices[start:end]
                if len(expert_idx) > 0:
                    expert = create_expert(expert_idx)
                    routed_experts_list.append(expert)
            
            routed_experts_module = nn.ModuleList(routed_experts_list)
            hidden_size = layer.mlp.gate_proj.in_features
            
            # Limpa VRAM antes de instanciar a estrutura complexa do MoE
            layer.mlp.to("cpu")
            del layer.mlp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Instancia o Novo Core DeepSliceMoE
            deepslice_moe = DeepSliceMoE(
                hidden_size=hidden_size,
                shared_expert=shared_expert,
                routed_experts=routed_experts_module,
                num_experts_per_tok=self.num_experts_per_tok
            ).to(self.device)
            
            layer.add_module("mlp", deepslice_moe)
            logger.info(f"Layer {layer_idx}: Convertida para DeepSliceMoE -> Shared: {n_shared} nerônios | {self.num_routed_experts} Experts de ~{n_per_expert} neurônios")
            
        return self.model
