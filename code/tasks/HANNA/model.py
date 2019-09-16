import sys

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ask_agent import AskAgent
from verbal_ask_agent import VerbalAskAgent


class LSTMWrapper(nn.Module):

    def __init__(self, input_size, hidden_size, dropout_ratio,
            device, batch_first=False):

        super(LSTMWrapper, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, 1,
            batch_first=batch_first,
            bidirectional=False)

        self.init_h = torch.zeros((1, 1, hidden_size),
            dtype=torch.float, device=device)
        self.device = device

    def init_state(self, batch_size):
        return (self.init_h.expand(-1, batch_size, -1).contiguous(),
                self.init_h.expand(-1, batch_size, -1).contiguous())

    def __call__(self, input, h):
        return self.lstm(input, h)


class EncoderLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
            dropout_ratio, device):

        super(EncoderLSTM, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = LSTMWrapper(
            embedding_size,
            hidden_size,
            dropout_ratio,
            device,
            batch_first=True)
        self.device = device

    def forward(self, input):

        embed = self.embedding(input)
        embed = self.drop(embed)

        init_state = self.lstm.init_state(input.size(0))
        ctx, _ = self.lstm(embed, init_state)

        """
        # Sort input by length
        sorted_lengths, forward_index_map = lengths.sort(0, True)

        input = input[forward_index_map]
        embed = self.embedding(input)   # (batch, seq_len, embedding_size)
        embed = self.drop(embed)
        init_state = self.lstm.init_state(input.size(0))

        packed_embed = pack_padded_sequence(embed, sorted_lengths,
            batch_first=True)
        enc_hs, _ = self.lstm(packed_embed, init_state)


        ctx, lengths = pad_packed_sequence(enc_hs, batch_first=True)
        ctx = self.drop(ctx)

        # Unsort outputs
        _, backward_index_map = forward_index_map.sort(0, False)
        ctx = ctx[backward_index_map]
        """

        return ctx


class Attention(nn.Module):

    def __init__(self, h_dim, v_dim, dot_dim=256):

        super(Attention, self).__init__()

        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_v = nn.Linear(v_dim, dot_dim, bias=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, query, values, mask=None):
        target = self.linear_in_h(query).unsqueeze(2)  # batch x dot_dim x 1
        context = self.linear_in_v(values)  # batch x v_num x dot_dim

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x v_num
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x v_num

        weighted_context = torch.bmm(attn3, values).squeeze(1)  # batch x v_dim

        return weighted_context, attn

class EltwiseProdScoring(nn.Module):

    def __init__(self, h_dim, a_dim, dot_dim):

        super(EltwiseProdScoring, self).__init__()

        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_a = nn.Linear(a_dim, dot_dim, bias=True)
        self.linear_out = nn.Linear(dot_dim, 1, bias=True)

    def forward(self, h, all_u_t, mask=None):
        target = self.linear_in_h(h).unsqueeze(1)  # batch x 1 x dot_dim
        context = self.linear_in_a(all_u_t)  # batch x a_num x dot_dim
        eltprod = torch.mul(target, context)  # batch x a_num x dot_dim
        logits = self.linear_out(eltprod).squeeze(2)  # batch x a_num
        return logits


class DecoderLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, dropout_ratio, device):

        super(DecoderLSTM, self).__init__()

        self.drop = nn.Dropout(p=dropout_ratio)

        self.lstm = LSTMWrapper(
            input_size,
            hidden_size,
            dropout_ratio,
            device)

        self.device = device

    def init_state(self, batch_size):
        self.zeros_vec = torch.zeros(
            (batch_size, 1), dtype=torch.uint8, device=self.device)
        return self.lstm.init_state(batch_size)

    def __call__(self, input, h):

        input_drop = self.drop(input)
        output, new_h = self.lstm(input_drop.unsqueeze(0), h)
        output = self.drop(output.squeeze(0))

        return output, new_h


class Seq2SeqModel(nn.Module):

    def __init__(self, vocab_size, decoder_input_size, agent_class, hparams,
            device):

        super(Seq2SeqModel, self).__init__()

        enc_hidden_size = hparams.hidden_size

        self.encoder = EncoderLSTM(
            vocab_size,
            hparams.word_embed_size,
            enc_hidden_size,
            hparams.instr_padding_idx,
            hparams.dropout_ratio,
            device)

        self.decoder = DecoderLSTM(
            decoder_input_size,
            hparams.hidden_size,
            hparams.dropout_ratio,
            device)

        self.curr_visual_attention = Attention(
            hparams.hidden_size, hparams.img_feature_size)

        self.next_visual_attention = Attention(
            hparams.hidden_size, hparams.img_feature_size)

        self.goal_visual_attention = Attention(
            hparams.hidden_size, hparams.img_feature_size)

        self.drop = nn.Dropout(p=hparams.dropout_ratio)

        self.text_attention = Attention(
            hparams.hidden_size, hparams.hidden_size)

        self.device = device

    def match_view(self, view_a, view_b):
        dot_product = torch.bmm(view_a, view_b.transpose(1, 2))
        view_a_norm = torch.norm(view_a, p=2, dim=2)
        view_b_norm = torch.norm(view_b, p=2, dim=2)
        norm_product = torch.bmm(
            view_a_norm.unsqueeze(2), view_b_norm.unsqueeze(1))
        norm_product = norm_product.clamp(min=1e-8)
        cosine_similarities = dot_product / norm_product
        d, _ = torch.max(cosine_similarities, dim=2)
        return d

    def step(self, h, inputs, text_ctx, text_ctx_mask, curr_view_features,
            goal_view_features):
        input = torch.cat(inputs, dim=1)
        output, h = self.decoder(input, h)
        c_text, _ = self.text_attention(output, text_ctx, text_ctx_mask)
        c_next, _ = self.next_visual_attention(c_text, curr_view_features)
        c_goal, _ = self.goal_visual_attention(c_text, goal_view_features)
        feature = torch.cat((output, c_text, c_next, c_goal), dim=1)
        return h, feature

    def init_state(self, batch_size):
        return self.decoder.init_state(batch_size)

    def encode(self, seq):
        return self.encoder(seq)


class NavModule(Seq2SeqModel):

    def __init__(self, vocab_size, agent_class, hparams, device):

        visual_feature_with_loc_size = \
            hparams.img_feature_size + hparams.loc_embed_size

        nav_input_size = visual_feature_with_loc_size + \
            hparams.img_feature_size + hparams.nav_dist_size

        super(NavModule, self).__init__(
            vocab_size, nav_input_size, agent_class, hparams, device)

        self.start_action = torch.zeros(
            visual_feature_with_loc_size, dtype=torch.float, device=device)

        self.predictor = EltwiseProdScoring(
            hparams.hidden_size * 2 + hparams.img_feature_size * 2,
            visual_feature_with_loc_size,
            hparams.hidden_size)

        self.device = device

    def init_action(self, batch_size):
        return self.start_action.expand(batch_size, -1)

    def decode(self, h, prev_a, a_embeds, text_ctx, text_ctx_mask,
            curr_view_features, goal_view_features, logit_mask):

        scene, _ = self.curr_visual_attention(h[0][-1], curr_view_features)
        goal_sim = self.match_view(curr_view_features, goal_view_features)

        inputs = (prev_a, scene, goal_sim)
        h, feature = self.step(h, inputs, text_ctx, text_ctx_mask,
            curr_view_features, goal_view_features)

        logit = self.predictor(feature, a_embeds)
        logit.data.masked_fill_(logit_mask, -float('inf'))

        return h, logit


class AskModule(Seq2SeqModel):

    def __init__(self, vocab_size, agent_class, hparams, device):

        visual_feature_with_loc_size = \
            hparams.img_feature_size + hparams.loc_embed_size

        ask_input_size = hparams.ask_embed_size + \
            hparams.img_feature_size + hparams.nav_dist_size * 2

        super(AskModule, self).__init__(
            vocab_size, ask_input_size, agent_class, hparams, device)

        self.action_embedding = nn.Embedding(
            agent_class.n_input_ask_actions(), hparams.ask_embed_size,
            agent_class.ask_actions.index('<start>'))

        self.predictor = nn.Linear(
            hparams.hidden_size * 2 + hparams.img_feature_size * 2,
            agent_class.n_output_ask_actions())

        self.start_action = torch.ones(1, dtype=torch.long, device=device) * \
            agent_class.ask_actions.index('<start>')

        self.device = device

    def init_action(self, batch_size):
        return self.start_action.expand(batch_size)

    def decode(self, h, prev_a, nav_dist, text_ctx, text_ctx_mask,
            curr_view_features, goal_view_features, logit_mask):

        prev_a = self.action_embedding(prev_a)
        scene, _ = self.curr_visual_attention(h[0][-1], curr_view_features)
        goal_sim = self.match_view(curr_view_features, goal_view_features)

        inputs = (prev_a, scene, goal_sim, nav_dist)
        h, feature = self.step(h, inputs, text_ctx, text_ctx_mask,
            curr_view_features, goal_view_features)

        logit = self.predictor(feature)
        logit.data.masked_fill_(logit_mask, -float('inf'))

        return h, logit


class AgentModel(nn.Module):

    def __init__(self, vocab_size, hparams, device):
        super(AgentModel, self).__init__()

        agent_class = AskAgent
        self.nav_module = NavModule(vocab_size, agent_class, hparams, device)
        self.ask_module = AskModule(vocab_size, agent_class, hparams, device)

    def init_state(self, batch_size):
        return self.nav_module.init_state(batch_size), \
               self.ask_module.init_state(batch_size)

    def init_action(self, batch_size):
        return self.nav_module.init_action(batch_size), \
               self.ask_module.init_action(batch_size)

    def encode(self, *args, **kwargs):
        return self.nav_module.encode(*args, **kwargs), \
               self.ask_module.encode(*args, **kwargs)

    def decode_nav(self, *args, **kwargs):
        return self.nav_module.decode(*args, **kwargs)

    def decode_ask(self, *args, **kwargs):
        return self.ask_module.decode(*args, **kwargs)





