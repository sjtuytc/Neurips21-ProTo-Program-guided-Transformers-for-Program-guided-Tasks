import torch.nn as nn
from mnnm_models.attention_modules import *


class Module(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head, encode_vision=False):
        super(Module, self).__init__()
        self.encode_vision = encode_vision
        self.prog_self_attention = SA(hidden_size, head_num, ff_size, dropout, hidden_size_head)
        if encode_vision:
            self.vision_self_attention = SA(hidden_size, head_num, ff_size, dropout, hidden_size_head)
        self.cross_attention = GA(hidden_size, head_num, ff_size, dropout, hidden_size_head)

    def forward(self, inputs, mask, vis_feat, vis_mask_tmp, program_masks, alpha):
        alpha = alpha.unsqueeze(-1)
        trans_mask = (1 - mask).unsqueeze(1).to(torch.bool)
        enc_output = self.prog_self_attention(inputs, trans_mask)
        enc_output = self.cross_attention(enc_output, vis_feat, vis_mask_tmp)
        if self.encode_vision:
            vis_feat = self.cross_attention(vis_feat, enc_output, None, None)
        enc_output = enc_output * program_masks.unsqueeze(-1)
        return alpha * enc_output + (1 - alpha) * inputs, vis_feat


class ShallowModule(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head, preprocessing=True):
        super(ShallowModule, self).__init__()
        # self.ffn = FFN(hidden_size, ff_size, dropout)
        # self.dropout = nn.Dropout(dropout, inplace=False)
        # self.norm = LayerNorm(hidden_size)
        self.cross_attention = GA(hidden_size, head_num, ff_size, dropout, hidden_size_head)

    def forward(self, inputs, vis_feat, vis_mask_tmp, program_masks):
        # inputs = self.norm(inputs + self.dropout(self.ffn(inputs)))
        enc_output = self.cross_attention(inputs, vis_feat, vis_mask_tmp)
        enc_output = enc_output * program_masks.unsqueeze(-1)
        return enc_output
