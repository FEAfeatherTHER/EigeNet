# Adapted from https://github.com/KyungsuKim42/tokensynth/blob/main/src/tokensynth/clap.py
import os
import numpy as np
import laion_clap
from pathlib import Path
import librosa
import torch
import torch.nn.functional as F
import warnings
from transformers import logging

class CLAP:
    """
    A class to handle audio and text embeddings using the CLAP model.
    
    Attributes:
        device (torch.device): The device (CPU or GPU) on which the model and tensors will be placed.
        clap (laion_clap.CLAP_Module): Instance of the LAION CLAP model.
    """
    def __init__(self, ckpt_path, device=None):
        """
        Initialize the CLAP class.
        Automatically selects GPU if available, otherwise CPU.
        Initializes the LAION CLAP model, loads the checkpoint, and sets the model to evaluation mode.
        
        Args:
            device (torch.device, optional): The desired device for the model. Defaults to None for auto-selection.
        """
        # Suppress non-critical warnings
        warnings.filterwarnings('ignore', message='torch.meshgrid:.*')
        # Disable transformers library's warnings
        logging.set_verbosity_error()
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the CLAP model
        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')

        self.clap.load_ckpt(ckpt_path)
        self.clap.to(self.device)
        
        # Set model to evaluation mode
        self.clap.eval()

    def encode_audio(self, audio, orig_sr):
        """
        Args:
            audio: [B, T], at orig_sr
            orig_sr: int
        
        Returns:
            torch.Tensor: A tensor containing the audio embedding.
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        if orig_sr != 48000:
            # audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=48000)
            audio = np.stack([librosa.resample(audio[i], orig_sr=orig_sr, target_sr=48000) for i in range(audio.shape[0])], axis=0)
        else:
            audio = audio
        # print(audio.shape)
        
        
        # Convert audio data to a tensor and move it to the device
        audio = torch.tensor(audio).float().to(self.device)
        
        
        # Get audio embedding from the CLAP model
        return self.clap.get_audio_embedding_from_data(audio, use_tensor=True)
    
    def encode_text(self, text):
        """
        Retrieve CLAP text embedding from a single string input.
        
        Args:
            text (str): A string containing the input text.
        
        Returns:
            torch.Tensor: A tensor containing the text embedding.
        
        Raises:
            AssertionError: If the provided text is not a string.
        """
        assert isinstance(text, str), "text must be a string"
        return self.clap.get_text_embedding([text], use_tensor=True)


if __name__ == "__main__":
    import os
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    clap = CLAP(ckpt_path="/data/250010171/ckpts/pretrained/clap/clap_music_audioset_epoch_15_esc_90.14.pt")
    audio_path = '/data/250010171/workspace/jingchong/data/debug/135.wav'
    audio1 = librosa.load(audio_path, sr=48000)[0]
    audios = np.stack([audio1, audio1, audio1], axis=0) # [3, T]
    print(audios.shape)
    clap_embed = clap.encode_audio(audios, 48000)
    print(clap_embed.shape)
    if torch.isinf(clap_embed).any() or torch.isnan(clap_embed).any():
        print("clap_embed is inf or nan")
    else:
        print("clap_embed is not inf or nan")