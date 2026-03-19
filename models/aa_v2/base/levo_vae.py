import json
import torch.nn as nn
import torch
from torch import Tensor
# from huggingface_hub import hf_hub_download
from stable_audio_tools.models.factory import create_model_from_config
# from stable_audio_tools.models.autoencoders import AudioAutoencoder


class LevoVAE(nn.Module):

    def __init__(self, device="cuda"):
        super().__init__()

        config_path = '/data/250010171/ckpts/pretrained/levo/vae/stable_audio_1920_vae.json'
        model_path = '/data/250010171/ckpts/pretrained/levo/vae/autoencoder_music_1320k.ckpt'
        with open(config_path, "r") as f:
            model_config = json.load(f)

        self.vae = create_model_from_config(model_config)
        ckpt = torch.load(model_path, map_location='cpu')
        load_result = self.vae.load_state_dict(ckpt["state_dict"], strict=False)

        for name, param in self.vae.named_parameters():
            param.requires_grad = False
        self.vae.to(device)
        self.vae.eval()

        self.sr = model_config["sample_rate"]
        self.fps = 25
    
    @torch.no_grad()
    def encode(self, audio: Tensor) -> Tensor:
        r"""

        Args:
            audio: (b, 2, l)
        """

        # audio_list = [a for a in audio]
        # audios = self.vae.preprocess_audio_list_for_encoder(audio_list, [self.sr] * len(audio_list))
        # latent = self.vae.encode_audio(audios)
        latent = self.vae.encode_audio(audio) # (B, 64, n_frames)
        return latent

    @torch.no_grad()
    def decode(self, latent):
        audio = self.vae.decode_audio(latent)
        return audio

    def __call__(self, audio: Tensor) -> Tensor:
        return self.encode(audio)


if __name__ == "__main__":
    import soundfile as sf
    import librosa
    import glob
    import os
    from tqdm import tqdm
    import numpy as np
    def encode_aduio(audio):
        latent = vae.encode(audio)
        return latent
    def decode_latent(latent):
        audio = vae.decode(latent)[0].transpose(-1,-2).cpu().numpy()
        return audio
    vae = LevoVAE("cuda")
    clip_id = 3131
    latent_path = f'/data/share/amphion/data/music_datasets/pre_latent/levovae/baby_fil/{clip_id}.npy'
    latent = np.load(latent_path)
    latent = torch.from_numpy(latent).unsqueeze(0).float().to("cuda")
    print(latent.shape)
    # wav = '/data/250010171/code/AnyTrainer-midi2audio/data/debug/0.wav'
    # wav = librosa.load(wav, sr=48000, mono=False)[0]
    # wav = torch.from_numpy(wav).unsqueeze(0).float().to("cuda")
    # latent = encode_aduio(wav)
    print(latent.shape)
    with torch.no_grad():
        audio = decode_latent(latent)
        print(audio.shape)
        tgt_path = f"/data/250010171/code/AnyTrainer-midi2audio/data/debug/recon_{clip_id}.wav"
        sf.write(tgt_path, audio, 48000)
        print(f"save to {tgt_path}")
        

                
    

