"""
Gemma 4 Neo — Configuration
Controla quais modificações estão ativas e os hiperparâmetros de cada uma.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# Layers globais (full-attention / Gated Attention)
GLOBAL_LAYER_INDICES = {5, 11, 17, 23, 29, 35, 41}


@dataclass
class ReLU2Config:
    """
    Configuração do MLP esparso com ReLU².

    ReLU²(x) = max(0, x)²  →  zeros exatos, distribuição mais esparsa que GELU/SiLU.
    O per_layer_input_gate (já existente no Gemma 4) é reutilizado como
    preditor de importância de neurônio: neurônios com score < threshold são pulados.
    """
    enabled: bool = True
    # Fração de neurônios MLP que podem ser zerados por forward pass.
    # 0.0 = desabilitado; 0.65 = 65% de zeros (valor calibrado empiricamente).
    sparsity_target: float = 0.65
    # Se True, usa o per_layer_input_gate como preditor de sparsidade.
    # Se False, aplica threshold global pela magnitude da pré-ativação.
    use_gate_as_predictor: bool = True
    # Número de neurônios ATIVOS (top-k do gate-score).
    # None → calculado automaticamente como int((1 - sparsity_target) * intermediate_size)
    topk_neurons: Optional[int] = None
    # Calibração: nomes de datasets HuggingFace para estimar threshold por layer.
    calibration_dataset: str = "wikitext"
    calibration_samples: int = 512


@dataclass
class GatedAttentionConfig:
    """
    Gated Attention (NeurIPS 2025 Best Paper — Qwen Team).

    Adiciona um gate σ(X · Wθ) por cabeça de atenção após o SDPA.
    O gate aprende a suprimir cabeças irrelevantes → elimina attention sinks,
    melhora extrapolação de contexto longo.

    Aplicado apenas aos 7 layers GLOBAIS (5, 11, 17, 23, 29, 35, 41).
    """
    enabled: bool = True
    # Inicialização: ones → gate começa transparente (comportamento = original).
    init_ones: bool = True
    # Aplicar SnapKV nestes layers para comprimir KV cache.
    snap_kv_enabled: bool = True
    # Quantas keys/values manter por cabeça (top-k por score de atenção acumulado).
    snap_kv_window: int = 1024  # tokens
    snap_kv_max_capacity: int = 4096  # máximo absoluto de tokens no KV


@dataclass
class Mamba2Config:
    """
    Configuração para substituição de SWA → Mamba-2 SSM.

    Esta é a modificação mais pesada — requer destilação.
    Os 35 layers locais (SWA, q=2048, k/v=512) são substituídos
    por blocos Mamba-2 com estado comprimido O(d_state).
    """
    enabled: bool = False  # Desabilitado por padrão — requer destilação
    d_state: int = 128          # Dimensão do estado SSM por cabeça
    d_conv: int = 4             # Kernel da convolução causal
    expand: int = 2             # Fator de expansão interno
    headdim: int = 64           # Dimensão por cabeça SSM
    chunk_size: int = 256       # Chunk size para algoritmo parallel scan
    # Caminho para checkpoint de Mamba-2 pré-destilado (None = inicializa do zero)
    pretrained_mamba_path: Optional[str] = None


@dataclass
class QuantizationConfig:
    """
    W4A8 quantização em cascata.

    - Pesos: INT4 via AWQ (asymmetric, group_size=128)
    - Ativações: INT8 dinâmico no residual stream
    - KV cache nos 7 layers globais: INT8
    """
    enabled: bool = True
    weight_bits: int = 4               # 4-bit pesos (NF4/INT4 AWQ)
    activation_bits: int = 8          # 8-bit ativações
    group_size: int = 128             # Group size para quantização de pesos
    kv_cache_bits: int = 8            # KV cache nos layers globais
    # Se True, usa bitsandbytes NF4 (mais simples, menos preciso)
    # Se False, usa AWQ (mais preciso, requer calibração)
    use_bnb: bool = True


@dataclass
class SpeculativeConfig:
    """
    Speculative decoding: E2B como draft → E4B/27B como verifier.
    """
    enabled: bool = False
    draft_model_id: str = "google/gemma-4-e2b-it"
    num_draft_tokens: int = 5          # Tokens propostos por step
    acceptance_threshold: float = 0.9  # Rejeitar se prob ratio < threshold


@dataclass
class Gemma4NeoConfig:
    """
    Configuração mestra do Gemma 4 Neo.
    Cada componente pode ser habilitado/desabilitado independentemente.
    """
    # Modelo base de onde carregar os pesos
    base_model_id: str = "google/gemma-4-e2b-it"

    # Parâmetros da arquitetura Gemma 4 (não mudar)
    hidden_size: int = 2560
    intermediate_size: int = 10240
    num_hidden_layers: int = 42
    num_attention_heads: int = 8      # heads locais
    num_key_value_heads: int = 4      # GQA locais
    num_global_heads: int = 16        # heads globais (q=4096)
    num_global_kv_heads: int = 8      # GQA globais
    gate_bottleneck_size: int = 256   # per_layer_input_gate bottleneck
    global_layers: List[int] = field(
        default_factory=lambda: sorted(GLOBAL_LAYER_INDICES)
    )

    # Sub-configurações de cada modificação
    relu2: ReLU2Config = field(default_factory=ReLU2Config)
    gated_attention: GatedAttentionConfig = field(default_factory=GatedAttentionConfig)
    mamba2: Mamba2Config = field(default_factory=Mamba2Config)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)

    # Onde salvar/carregar o modelo Neo convertido
    output_dir: str = "./gemma4_neo_checkpoint"
    # Precisão de trabalho durante a conversão
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"

    def active_modifications(self) -> List[str]:
        active = []
        if self.relu2.enabled:
            active.append("ReLU² MLP + SparsityPredictor")
        if self.gated_attention.enabled:
            active.append("Gated Attention (global layers)")
        if self.mamba2.enabled:
            active.append("Mamba-2 SSM (local layers)")
        if self.quantization.enabled:
            active.append(f"W{self.quantization.weight_bits}A{self.quantization.activation_bits}")
        if self.speculative.enabled:
            active.append("Speculative Decoding")
        return active
