import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
from torch import TensorType
from typing import Optional, Tuple, Dict, Any
from transformers import AutoModel, AutoConfig, Dinov2Model, Dinov2Config


class DinoV3Encoder(nn.Module):
    """
    Args:
        model_name_or_path: HuggingFace model name or local path
        output_dim: output feature dimension
        frozen: whether to freeze model weights
        from_scratch: whether to initialize from scratch (not using pretrained weights)
        in_channels: input channels number (default 3)
    """
    
    def __init__(
        self,
        model_name_or_path: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        output_dim: int = 512,
        frozen: bool = True,
        from_scratch: bool = False,
        in_channels: int = 3,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.model_name = model_name_or_path
        
        assert "dino" in model_name_or_path.lower(), \
            "DinoV3Encoder only supports DINO models"
        assert in_channels == 3, \
            "DinoV3Encoder currently only supports 3-channel input"
        
        # check if it is a local path
        local_path = model_name_or_path
        is_local = os.path.exists(model_name_or_path)

        # load model
        if from_scratch:
            config = AutoConfig.from_pretrained(local_path, local_files_only=is_local)
            self.model = AutoModel.from_config(config)
            print(f"DinoV3Encoder: Initialized {model_name_or_path} from scratch (no pretrained weights)")
        else:
            self.model = AutoModel.from_pretrained(
                local_path, 
                local_files_only=is_local,
                #device_map="auto", # delete this line to avoid memory error
            )
            print(f"DinoV3Encoder: Loaded pretrained {model_name_or_path} (local={is_local})")
        
        # decide whether to freeze based on frozen parameter
        if frozen:
            self._freeze()
        
        # get model hidden layer dimension
        hidden_size = self.model.config.hidden_size
        
        # output projection layer
        if output_dim != hidden_size:
            self.proj = nn.Linear(hidden_size, output_dim, bias=False)
        else:
            self.proj = nn.Identity()
        
        # print trainable parameters statistics
        self._print_trainable_params()
    
    def _freeze(self):
        """freeze all model parameters"""
        for n, p in self.model.named_parameters():
            # LayerNorm layer remains trainable (optional), other layers are frozen
            p.requires_grad = "LayerNorm" in n.split(".")
    
    def _print_trainable_params(self):
        """print trainable parameters number"""
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_trainable += sum(p.numel() for p in self.proj.parameters() if p.requires_grad)
        print(f"DinoV3Encoder [{self.model_name}] - Trainable params: {n_trainable / 1e6:.2f}M")
    
    def forward(self, x: TensorType) -> TensorType:
        """
        forward propagation.
        
        Args:
            x: input tensor, shape [B, C, H, W] or [B, 3, H, W]
               depth map C=3 represents x, y, z components of 3D coordinates
            
        Returns:
            pooled_output: pooled features, shape [B, output_dim]
        """
        inputs = {'pixel_values': x.to(self.model.device)}
        outputs = self.model(**inputs)
        pooled_output = outputs.pooler_output
        pooled_output = self.proj(pooled_output)
        return pooled_output
    
    def forward_with_intermediates(self, x: TensorType) -> Dict[str, Any]:
        """
        return intermediate layer features (for feature pyramid etc.).
        
        Args:
            x: input tensor, shape [B, C, H, W]
            
        Returns:
            dict: contains pooler_output and hidden_states etc. intermediate features
        """
        inputs = {'pixel_values': x.to(self.model.device)}
        outputs = self.model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states  # tuple of all hidden states
        
        return {
            'pooled_output': self.proj(outputs.pooler_output),
            'last_hidden_state': outputs.last_hidden_state,
            'hidden_states': hidden_states,
            'num_layers': len(hidden_states),
        }
    
    def get_intermediate_features(
        self, 
        x: TensorType, 
        layer_indices: Optional[list] = None
    ) -> Tuple[TensorType, list]:
        """
        get features of specified intermediate layers.
        
        Args:
            x: input tensor
            layer_indices: list of layer indices to get, None represents all layers
            
        Returns:
            features: concatenated intermediate layer features
            layer_idx_list: corresponding layer indices
        """
        inputs = {'pixel_values': x.to(self.model.device)}
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        if layer_indices is None:
            # default return last 4 layers
            layer_indices = list(range(max(1, len(hidden_states) - 4), len(hidden_states)))
        
        selected = [hidden_states[i] for i in layer_indices]
        # pool each layer and concatenate
        features = torch.cat([h.mean(dim=1) for h in selected], dim=-1)
        return features, layer_indices
    
    def unlock(self, unlocked_layers: int = 0):
        """
        unlock model weights.
        
        Args:
            unlocked_layers: unlock last N layers, 0 represents all layers
        """
        if unlocked_layers <= 0:
            for p in self.model.parameters():
                p.requires_grad = True
        else:
            # only unlock last N layers
            num_layers = len(list(self.model.parameters()))
            for i, (n, p) in enumerate(self.model.named_parameters()):
                if i >= num_layers - unlocked_layers * 12:  # assume each layer has about 12 parameters
                    p.requires_grad = True
    
    @property
    def device(self):
        """get model device"""
        return self.model.device
    
    @property
    def dtype(self):
        """get model data type"""
        return next(self.model.parameters()).dtype


def create_dinov3(
    model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
    output_dim: int = 512,
    frozen: bool = True,
    from_scratch: bool = False,
    in_channels: int = 3,
    device: str = "cuda",
) -> DinoV3Encoder:
    """
    factory function to create dinov3 encoder.
    
    Args:
        model_name: HuggingFace model name
        output_dim: output feature dimension
        frozen: whether to freeze weights
        from_scratch: whether to initialize from scratch
        in_channels: input channels number
        device: device "cuda" or "cpu"
        
    Returns:
        DinoV3Encoder instance
    """
    model = DinoV3Encoder(
        model_name_or_path=model_name,
        output_dim=output_dim,
        frozen=frozen,
        from_scratch=from_scratch,
        in_channels=in_channels,
    )
    model = model.to(device)
    return model


def create_dinov3_from_config(config: dict, device: str = "cuda") -> DinoV3Encoder:
    """
    create dinov3 encoder from configuration dictionary (compatible with AGREE's vision_cfg format).
    
    Args:
        config: dictionary containing the following keys:
            - hf_model_name: HuggingFace model name
            - output_dim: output dimension (optional, default 512)
            - frozen: whether to freeze (optional, default True)
            - from_scratch: whether to initialize from scratch (optional, default False)
            - in_channels: input channels number (optional, default 3)
        device: device "cuda" or "cpu"
        
    Returns:
        DinoV3Encoder instance
    """
    return create_dinov3(
        model_name=config.get("hf_model_name", "facebook/dinov3-vits16-pretrain-lvd1689m"),
        output_dim=config.get("embed_dim", 512),
        frozen=config.get("frozen", True),
        from_scratch=config.get("from_scratch", False),
        in_channels=config.get("in_channels", 3),
        device=device,
    )


# ============== example ==============
if __name__ == "__main__":
    # local model path
    LOCAL_MODEL_PATH = "/data/250010171/ckpts/pretrained/dinov3-vits16-pretrain-lvd1689m"

    # 1. basic usage - load pretrained dinov3 from local path
    print("=" * 50)
    print("create dinov3 encoder (local path)")
    encoder = create_dinov3(
        model_name=LOCAL_MODEL_PATH,
        output_dim=512,
        frozen=False, #finetune the model
    )
    
    # 2. test forward propagation
    dummy_input = torch.randn(2, 3, 256, 512)  # [B, 3, H, W]
    output = encoder(dummy_input)
    print(f"input shape: {dummy_input.shape}")
    print(f"output shape: {output.shape}")
    
    # 3. get intermediate layer features

    features_dict = encoder.forward_with_intermediates(dummy_input)
    print(f"pooled output shape: {features_dict['pooled_output'].shape}")
    print(f"last hidden layer shape: {features_dict['last_hidden_state'].shape}")
    print(f"total layers: {features_dict['num_layers']}")
    