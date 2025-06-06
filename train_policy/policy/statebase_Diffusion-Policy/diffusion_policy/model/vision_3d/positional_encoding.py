import torch
import numpy as np

class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, 1, -1, 1))

    def forward(self, x):
        """
        x: B,N,self.d_in
        out: B,N,self.d_out
        """
        embed = x.unsqueeze(-2)
        embed = embed.repeat(1, 1, self.num_freqs * 2, 1)
        embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs)) # B,N,2*num_freqs,d_in
        embed = embed.reshape(*x.shape[:-1], -1)
        if self.include_input:
            embed = torch.cat((x, embed), dim=-1)
        return embed

if __name__ == "__main__":
    point_cloud = torch.rand(64, 512, 3)
    
    pos_encoder = PositionalEncoding(d_in=3)
    
    pos_encoded = pos_encoder(point_cloud)
    print(pos_encoded.shape)
    