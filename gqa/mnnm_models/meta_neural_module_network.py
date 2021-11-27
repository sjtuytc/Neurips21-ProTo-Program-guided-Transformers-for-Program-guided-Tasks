import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import math
from gqa_mnnm_constants import *
import numpy as np
from mnnm_models.general_deep_modules import FC, MLP, LayerNorm
from mnnm_models.attention_modules import *
from mnnm_models.meta_modules import *


class TreeTransformerSparsePostv2(nn.Module):
    def __init__(self, vocab_size, answer_size, visual_dim, hidden_dim, coordinate_dim,
                 n_head, n_layers, stacking, dropout, intermediate_dim, pre_layers, intermediate_layer):
        super(TreeTransformerSparsePostv2, self).__init__()
        self.forward_ques = True

        # The question encoder
        self.embedding = nn.Embedding(vocab_size, 300, padding_idx=PAD)
        self.ques_proj = nn.Linear(300, hidden_dim)
        self.prog_proj = nn.Linear(300, hidden_dim // 8)

        self.ques_pos_emb = PositionalEmbedding(hidden_dim)
        self.intermediate_layer = intermediate_layer

        # The visual encoder
        self.vis_proj = nn.Linear(visual_dim, hidden_dim)
        self.coordinate_proj = nn.Linear(coordinate_dim, hidden_dim)
        # self.vis_encoder = VisualEncoder(hidden_dim, n_head, pre_layers, dropout)
        self.ques_encoder = nn.ModuleList(
            [SA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
             for _ in range(pre_layers)])
        self.vis_encoder = nn.ModuleList(
            [SGA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
             for _ in range(pre_layers)])

        # The program decoder
        self.num_regions = intermediate_dim
        self.pos_emb = nn.Embedding(self.num_regions, hidden_dim)
        self.idx_predictor = nn.Linear(hidden_dim, self.num_regions)

        # The self attention module beforehand
        self.post = nn.ModuleList([ShallowModule(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
                                   for _ in range(stacking)])

        # The self attention module and cross attention module
        self.module = Module(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)

        # Projection layer to retrieve final answer
        self.proj = nn.Linear(hidden_dim, answer_size) if not self.forward_ques else nn.Linear(hidden_dim * 2, answer_size)

    def forward(self, ques, ques_masks, program, program_masks, transition_masks, activate_masks, vis_feat, box_feat,
                vis_mask, index, depth, intermediate_idx=None):
        # ques: [BS, QD], ques_masks: [BS, QD], program: [BS, T, PF], program_masks: [BS, R],
        # transition_masks: [BS, LAYER, R, R], activate_masks: [BS, LAYER, R], vis_feat: [BS, N, D],
        # box_feat: [BS, N, 6], vis_mask: [BS, N], index: [BS], depth: [BS, R], intermediate_idx: [BS, R, N]
        batch_size = ques.size(0)
        length = program.size(1)
        idx = torch.arange(vis_feat.size(1)).unsqueeze(0).repeat(batch_size, 1).to(ques.device)
        vis_feat = self.vis_proj(vis_feat) + self.coordinate_proj(box_feat) + self.pos_emb(idx)

        vis_mask_tmp = (1 - vis_mask).unsqueeze(1).unsqueeze(2).to(torch.bool)
        program_mask_tmp = (1 - program_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)
        ques_mask_tmp = (1 - ques_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)

        # Encoding the question with self-attention
        ques_input = self.ques_proj(self.embedding(ques)) + self.ques_pos_emb(ques)
        for enc in self.ques_encoder:
            ques_input = enc(ques_input, ques_mask_tmp)
            ques_input *= ques_masks.unsqueeze(-1)

        # Encoding the visual feature
        for enc in self.vis_encoder:
            vis_feat = enc(vis_feat, ques_input, vis_mask_tmp, ques_mask_tmp)
            vis_feat *= vis_mask.unsqueeze(-1)

        start_enc_output = self.prog_proj(self.embedding(program)).view(batch_size, program.size(1), -1)  # [BS, R, D]
        transition_masks = transition_masks.transpose(0, 1)  # [T, BS, R, R]
        activate_masks = activate_masks.transpose(0, 1)  # [T, BS, R]

        enc_output = start_enc_output

        # Build the structure into the transformer
        for trans_mask, active_mask in zip(transition_masks, activate_masks):
            enc_output, vis_feat = self.module(enc_output, trans_mask, vis_feat, vis_mask_tmp, program_masks, active_mask)
            # trans_mask: [BS, R, R], vis_feat: [BS, N, D],
            # vis_mask_tmp: [BS, 1, 1, D], program_mask: [BS, R], active_mask: [BS, R]

        # Post-Processing the encoder output
        for layer in self.post:
            enc_output = layer(enc_output, vis_feat, vis_mask_tmp, program_masks)

        # Predict the intermediate results
        pre_logits = self.idx_predictor(enc_output)
        lang_feat = torch.gather(enc_output, 1, index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, enc_output.size(-1)))
        lang_feat = lang_feat.view(batch_size, -1)
        if self.forward_ques:
            ques_feat = torch.mean(ques_input, dim=1)
            logits = self.proj(torch.cat([lang_feat, ques_feat], dim=-1))
        else:
            logits = self.proj(lang_feat)

        # pre_logits: [BS, P, N], logits:[BS, A]
        return pre_logits, logits


if __name__ == '__main__':
    test_m = TreeTransformerSparsePostv2(vocab_size=3762, stacking=2, answer_size=1845, visual_dim=2048,
                                         coordinate_dim=6, hidden_dim=512, n_head=8, n_layers=5, dropout=0.1,
                                         intermediate_dim=49, pre_layers=3, intermediate_layer=False)
    # ques, ques_masks, prog, prog_masks, trans_masks, activate_masks, object_feat, box_feat, vis_mask, index, depth = \
    #     torch.tensor()
