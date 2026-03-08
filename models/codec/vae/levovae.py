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

        # config_path = hf_hub_download(
        #     repo_id="tencent/SongGeneration", 
        #     filename="ckpt/vae/stable_audio_1920_vae.json"
        # )

        # model_path = hf_hub_download(
        #     repo_id="tencent/SongGeneration", 
        #     filename="ckpt/vae/autoencoder_music_1320k.ckpt"

        config_path = '/data/250010171/ckpts/pretrained/levo/vae/stable_audio_1920_vae.json'
        model_path = '/data/250010171/ckpts/pretrained/levo/vae/autoencoder_music_1320k.ckpt'
        with open(config_path, "r") as f:
            model_config = json.load(f)

        self.vae = create_model_from_config(model_config)
        ckpt = torch.load(model_path, map_location='cpu')
        load_result = self.vae.load_state_dict(ckpt["state_dict"], strict=False)

        # print("Missing keys:", load_result.missing_keys)
        # print("Unexpected keys:", load_result.unexpected_keys)

        # if len(load_result.missing_keys) == 0 and len(load_result.unexpected_keys) == 0:
        #     print("✅ Model weights loaded successfully.")
        # else:
        #     print("⚠️ Some weights were not loaded completely!")
        
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
        latent = self.vae.encode_audio(audio)
        return latent

    @torch.no_grad()
    def decode(self, latent):
        audio = self.vae.decode_audio(latent)
        return audio

    def __call__(self, audio: Tensor) -> Tensor:
        return self.encode(audio)


# if __name__ == "__main__":
#     vae = LevoVAE()
#     audio = torch.randn(1, 2, 48000).to("cuda")
#     latent = vae.encode(audio)
#     print(latent.shape) # [1, 64, 25]
#     audio = vae.decode(latent)
#     print(audio.shape) # [1, 2, 48000]

#     import torchaudio
#     file_path = "/data/250010171/code/levo-vae/sayang-cut-3s.wav"
#     import librosa
#     waveform, sample_rate = librosa.load(file_path, sr=48000, mono=False)
#     waveform = torch.from_numpy(waveform).float().to("cuda")
#     waveform = waveform.unsqueeze(0)
#     latent = vae.encode(waveform)
#     print(latent.shape)
#     audio = vae.decode(latent)
#     print(audio.shape)
#     torchaudio.save("sayang-cut-3s-recon.wav", audio.squeeze(0).cpu(), 48000)

# if __name__ == "__main__":
#     import librosa
#     import numpy as np
#     import soundfile as sf
#     vae = LevoVAE(device="cuda")
#     audio_path = '/data/250010171/code/levo-vae/sayang-cut.wav'
#     audio, sr = librosa.load(audio_path, sr=48000, mono=False)
#     audio = torch.Tensor(audio).unsqueeze(0).to("cuda")
#     # audio = torch.Tensor(audio).unsqueeze(0)
#     # audio = audio.repeat(1, 2, 1)
#     with torch.no_grad():
#         latent = vae.encode(audio)
#         print(latent.shape)
#         new_audio = vae.decode(latent)
#         print(new_audio.shape) # [1, 2, T]
#         sf.write("sayang-cut-recon.wav", new_audio.squeeze(0).cpu().numpy().T, sr)
if __name__ == "__main__":
    import time
    vae = LevoVAE(device="cuda")
    audio = torch.randn(10, 2, 1426560).to("cuda")
    start_time = time.time()
    latent = vae.encode(audio)
    end_time = time.time()
    print("vae encode time: ", end_time - start_time)