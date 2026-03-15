import dac
import torch.nn as nn
import torch

class DAC(nn.Module):
    def __init__(self, path):
        super(DAC, self).__init__()
        self.model = dac.DAC.load(path)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def encode(self, x):
        """
        x : (b, 1, t)
        z: (b, 1024, t)
        codes: (b, 12, t)
        latents: (b, 96, t)
        """
        z, codes, latents, code_loss, commit_loss = self.model.encode(x)
        # print(f"*"*20)
        # print(f"z: {z.shape}")
        # print(f"codes: {codes.shape}")
        # print(f"latents: {latents.shape}")
        # print(f"code_loss: {code_loss}")
        # print(f"commit_loss: {commit_loss}")
        return codes
    
    def encode_z(self, x):
        """
        x : (b, 1, t)
        z: (b, 1024, t)
        codes: (b, 12, t)
        latents: (b, 96, t)
        """
        z, codes, latents, code_loss, commit_loss = self.model.encode(x)
        # print(f"*"*20)
        # print(f"z: {z.shape}")
        # print(f"codes: {codes.shape}")
        # print(f"latents: {latents.shape}")
        # print(f"code_loss: {code_loss}")
        # print(f"commit_loss: {commit_loss}")
        return z

    
    def decode(self, codes):
        """
        self.model.quantizer.from_codes(codes) returns:
            z: (b, 1024, t)
            latents: (b, 96, t)
            codes: (b, 12, t)
        """
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
    bsz = 48 * 9
    device = "cuda"
    signal = torch.randn(bsz, 1, 8000).to(device)
    dac = DAC("/mnt/workspace/jingchong/ckpt/pretained/dac/ckpt/weights_16k.pth")
    dac = dac.to(device)
    import time
    start_time = time.time()
    codes = dac.encode(signal)
    end_time = time.time()
    print(f"encode time: {end_time - start_time}")
    print(f"codes: {codes.shape}")
    decoded = dac.decode(codes)
    print(f"decoded: {decoded.shape}")