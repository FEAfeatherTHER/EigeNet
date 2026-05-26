import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class InfoNCELoss(nn.Module):
    def __init__(self, init_tau=0.07, clamp_min=0, clamp_max=4.6052):
        super().__init__()
        # logit_scale = ln(1/tau)
        # tau = 0.07 , same as CLIP's default initial value
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_tau))
        
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x, y):
        """
        args:
            x : (B, D)

        """
        with torch.no_grad():
            self.logit_scale.clamp_(self.clamp_min, self.clamp_max)

        # 1. l2 normalize
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        # 2. compute similarity matrix and scale
        # logits: (B, B)
        # exp(logit_scale) = 1/tau
        t = self.logit_scale.exp()
        logits = torch.matmul(x, y.t()) * t

        # 3. label
        batch_size = x.size(0)
        labels = torch.arange(batch_size, device=x.device)

        # 4. compute symmetric cross entropy
        loss_x = F.cross_entropy(logits, labels)
        loss_y = F.cross_entropy(logits.t(), labels)

        return (loss_x + loss_y) / 2

    @property
    def current_tau(self):
        #for monitoring the current temperature value in the log
        return (1.0 / self.logit_scale.exp()).item()

if __name__ == "__main__":
    bsz = 64 * 9
    x = torch.randn(bsz, 128)
    y = torch.randn(bsz, 128)
    loss = InfoNCELoss()
    loss_value = loss(x, y)
    print(loss_value)