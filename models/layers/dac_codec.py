import dac
import torch.nn as nn
import torch
from einops import rearrange
class DAC(nn.Module):
    def __init__(self, path):
        super(DAC, self).__init__()
        self.model = dac.DAC.load(path)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def encode(self, x, layer = None):
        """
        x : (b, 1, t)
        z: (b, 1024, t)
        codes: (b, 12, t)
        latents: (b, 96, t)
        """
        z, codes, latents, code_loss, commit_loss = self.model.encode(x)
        depth = codes.shape[1]
        if layer is None:
            layer = depth
        else:
            layer = min(layer, depth)
        codes = codes[:, :layer]
        return codes
    
    def encode_latent(self, x, layer = None):
        """
        x : (b, 1, t)
        z: (b, 1024, t)
        codes: (b, 12, t)
        latents: (b, 8 * layer, t)
        """
        z, codes, latents, code_loss, commit_loss = self.model.encode(x)
        depth = codes.shape[1]
        if layer is None:
            layer = depth
        else:
            layer = min(layer, depth)
        latents = latents[:, :layer * 8]#(b , layer * 8, t)
        return latents
    
    def sum_latent_residual(self, latents, low_dim = 8):
        """
        latents: (b, layer * 8, t)
        return: (b, 8, t)
        """
        b, D, t = latents.shape
        d = D // low_dim
        latents = rearrange(latents, 'b (d c) t -> b c d t', c = low_dim)
        latents = torch.sum(latents, dim = 2)
        return latents
    
    def encode_z(self, x):
        """
        x : (b, 1, t)
        z: (b, 1024, t)
        codes: (b, 12, t)
        latents: (b, 96, t)
        """
        
        z, codes, latents, code_loss, commit_loss = self.model.encode(x)
        return z


    def decode(self, codes, layer = None):
        """
        self.model.quantizer.from_codes(codes) returns:
            z: (b, 1024, t)
            latents: (b, 96, t)
            codes: (b, 12, t)
        """
        depth = codes.shape[1]
        if layer is None:
            layer = depth
        else:
            layer = min(layer, depth)
        codes = codes[:, :layer]
        z, latents, codes = self.model.quantizer.from_codes(codes)
        # print(f"*"*20)
        # print(f"z: {z.shape}")
        # print(f"codes: {codes.shape}")
        # print(f"latents: {latents.shape}")
        audio = self.model.decode(z)
        return audio
    
    def decode_z(self, z):
        """
        self.model.quantizer.from_codes(codes) returns:
            z: (b, 1024, t)
            latents: (b, 96, t)
            codes: (b, 12, t)
        """
        audio = self.model.decode(z)
        return audio

if __name__ == "__main__":
    device = "cuda"
    signal_path = '/data/250010171/code/EigeNet_discriminant/data/debug/她说_68.63_73.63.wav'
    import librosa
    signal, sr = librosa.load(signal_path, sr = 16000)
    signal = torch.from_numpy(signal).unsqueeze(0).unsqueeze(0).to(device)
    dac = DAC("/data/250010171/ckpts/pretrained/dac/weights_16k.pth")
    dac = dac.to(device)
    codes = dac.encode(signal)
    print(f"codes: {codes.shape}")
    l = 12
    decoded = dac.decode(codes, layer = l)
    print(f"decoded: {decoded.shape}")
    import soundfile as sf
    recon_audio = decoded.squeeze(0).squeeze(0).cpu().numpy()
    sf.write(f'/data/250010171/code/EigeNet_discriminant/data/debug/recon_audio_{l}.wav', recon_audio, sr)
    