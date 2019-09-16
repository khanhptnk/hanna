import sys
import math
import copy
import functools

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ask_agent import AskAgent
from verbal_ask_agent import VerbalAskAgent
from attention import Attention, MultiHeadedAttention, SimAttention
from oracle import AskTeacher

dc = copy.deepcopy
clone = lambda module, n: [dc(module) for _ in range(n)]


class LayerNormResidual(nn.Module):
    '''
        Apply a generic transformation followed by dropout, residual connection,
            and layer normalization
    '''

    def __init__(self, module, input_size, output_size, dropout):

        super(LayerNormResidual, self).__init__()

        self.module = module
        self.layer_norm = nn.LayerNorm(output_size, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)

        if input_size != output_size:
            self.shortcut_layer = nn.Linear(input_size, output_size)
        else:
            self.shortcut_layer = lambda x: x

    def forward(self, input, *args, **kwargs):
        input_shortcut = self.shortcut_layer(input)
        return self.layer_norm(
            input_shortcut + self.dropout(self.module(input, *args, **kwargs)))


class PositionalEncoding(nn.Module):
    '''
        Sinusoid positional encoding. Implementation adapted from OpenNMT-py:
           https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/embeddings.py#L11
    '''

    def __init__(self, dim, max_len):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dim = dim

    def forward(self, emb, indices=None):
        if indices is None:
            emb = emb + self.pe[:, :emb.size(1)]
        else:
            emb = emb + self.pe.squeeze(0).index_select(0, indices)
        return emb


class LearnedTime(nn.Module):
    '''
        ResNet-based positional encoding
    '''

    def __init__(self, input_size, output_size):

        super(LearnedTime, self).__init__()
        module = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU()
        )
        self.increment_op = LayerNormResidual(
            module, input_size, output_size, 0)

        init_time = torch.zeros((output_size,), dtype=torch.float)
        self.register_buffer('init_time', init_time)

    def reset(self, batch_size):
        self.time = self.init_time.expand(batch_size, -1)
        return self.time

    def forward(self, other_time=None):
        if other_time is None:
            input_time = self.time
        else:
            input_time = torch.cat((self.time, other_time), dim=-1)
        self.time = self.increment_op(input_time)
        return self.time


class TransformerLayer(nn.Module):
    '''
        Generic Transformer layer
    '''

    def __init__(self, attention_heads, dropout):

        layer_norm_residual_fn = lambda module, input_size, output_size: \
            LayerNormResidual(module, input_size, output_size, dropout)

        self.attention_fn = lambda query_size, mem_size, output_size: \
            layer_norm_residual_fn(MultiHeadedAttention(
                query_size, mem_size, mem_size, output_size, attention_heads),
                query_size, output_size)

        self.feed_forward_fn = lambda input_size, output_size: \
            layer_norm_residual_fn(
                nn.Sequential(
                    nn.Linear(input_size, output_size * 4),
                    nn.ReLU(),
                    nn.Linear(output_size * 4, output_size)
                ),
                input_size, output_size)

        super(TransformerLayer, self).__init__()


class EncoderLayer(TransformerLayer):
    '''
        Text-encoding layer
    '''

    def __init__(self, query_size, mem_size, output_size, attention_heads, dropout):

        super(EncoderLayer, self).__init__(attention_heads, dropout)

        self.self_attention = self.attention_fn(
            query_size, mem_size, output_size)
        self.feed_forward   = self.feed_forward_fn(output_size, output_size)

    def forward(self, input, key, value, mask=None):

        output = self.self_attention(input, key, value, mask)
        output = self.feed_forward(output)

        return output


class TextDecoderLayer(TransformerLayer):
    '''
        Inter-task layer
    '''

    def __init__(self, query_size, mem_size, output_size,
            attention_heads, dropout):

        super(TextDecoderLayer, self).__init__(attention_heads, dropout)

        self.self_attention = self.attention_fn(
            query_size, mem_size, output_size)
        self.enc_attention  = self.attention_fn(
            output_size, mem_size, output_size)
        self.feed_forward = self.feed_forward_fn(output_size, output_size)

    def forward(self, input, self_key, self_value, enc_mem, enc_mask=None):

        hidden = self.self_attention(input, self_key, self_value)
        output = self.enc_attention(hidden, enc_mem, enc_mem, enc_mask)
        output = self.feed_forward(output)

        return hidden, output


class StateDecoderLayer(TransformerLayer):
    '''
        Intra-task layer
    '''

    def __init__(self, query_size, mem_size, output_size,
            attention_heads, dropout, hparams):

        super(StateDecoderLayer, self).__init__(attention_heads, dropout)

        self.self_attention = self.attention_fn(
            query_size, mem_size, output_size)

        self.no_sim_attend = hparams.no_sim_attend
        if self.no_sim_attend:
            self.feed_forward = self.feed_forward_fn(output_size, output_size)
        else:
            self.sim_attention = SimAttention()
            self.feed_forward = self.feed_forward_fn(output_size, output_size)
            self.gate = nn.Sequential(
                nn.Linear(output_size * 2, output_size),
                nn.Sigmoid()
            )

    def forward(self, input, key, value, mask=None):

        if self.no_sim_attend:
            output = self.self_attention(input, key, value)
            output = self.feed_forward(output)
        else:
            self_hidden = self.self_attention(input, key, value)
            self_hidden = self.feed_forward(self_hidden)
            sim_hidden  = self.sim_attention(input, key, value, mask=mask)
            beta = self.gate(torch.cat((self_hidden, sim_hidden), dim=-1))
            output = self_hidden - beta * sim_hidden

        return output


class Encoder(nn.Module):
    '''
        Text encoder
    '''

    def __init__(self, vocab_size, embed_size, hidden_size, padding_idx,
            max_instr_len, layer_fn, num_layers, dropout):

        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx)
        self.positional_encoding = PositionalEncoding(embed_size, max_instr_len)
        self.dropout = nn.Dropout(p=dropout)

        first_layer  = layer_fn(embed_size, embed_size)
        hidden_layer = layer_fn(hidden_size, hidden_size)
        self.layers = nn.ModuleList(
            [first_layer] + clone(hidden_layer, num_layers - 1))

    def forward(self, input, *args):

        input = self.embedding(input)
        input = input * math.sqrt(input.size(-1))
        output = self.dropout(self.positional_encoding(input))

        for layer in self.layers:
            output = layer(output, output, output, *args)

        return output


class Decoder(nn.Module):
    '''
        Generic decoder (can be used as both inter and intra decoders)
    '''

    def __init__(self, input_size, hidden_size, layer_fn, num_layers, dropout):

        super(Decoder, self).__init__()

        self.embedding = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)

        first_layer  = layer_fn(hidden_size, hidden_size)
        hidden_layer = layer_fn(hidden_size, hidden_size)
        self.layers = nn.ModuleList(
            [first_layer] + clone(hidden_layer, num_layers - 1))

        init_h = torch.zeros((hidden_size,), dtype=torch.float)
        self.register_buffer('init_h', init_h)

    def reset(self, batch_size):
        self.key   = [[self.init_h.expand(batch_size, -1).contiguous()]
            for _ in range(len(self.layers))]
        self.value = [[self.init_h.expand(batch_size, -1).contiguous()]
            for _ in range(len(self.layers))]

    def forward(self, input, time, *args, **kwargs):

        input = self.dropout(self.embedding(input) + time)

        for i, layer in enumerate(self.layers):

            key   = torch.stack(self.key[i]).transpose(0, 1).contiguous()
            value = torch.stack(self.value[i]).transpose(0, 1).contiguous()

            output = layer(input, key, value, *args, **kwargs)

            self.key[i].append(input)
            if isinstance(output, tuple):
                self.value[i].append(output[0])
                input = output[1]
            else:
                self.value[i].append(output)
                input = output

        return output


class Seq2SeqModel(nn.Module):

    def __init__(self, vocab_size, input_size, state_size, hparams, device):

        super(Seq2SeqModel, self).__init__()

        hidden_size = hparams.hidden_size
        dropout = hparams.dropout_ratio
        num_layers = hparams.num_layers
        attention_heads = hparams.attention_heads

        encoder_layer_fn = lambda input_size, mem_size: EncoderLayer(
            input_size, mem_size, hidden_size, attention_heads, dropout)

        self.encoder = Encoder(
            vocab_size,
            hparams.word_embed_size,
            hidden_size,
            hparams.instr_padding_idx,
            hparams.max_instr_len,
            encoder_layer_fn,
            num_layers,
            dropout)

        # Inter-task decoder
        text_decoder_layer_fn = lambda input_size, mem_size: TextDecoderLayer(
            input_size, mem_size, hidden_size, attention_heads, dropout)

        self.text_decoder = Decoder(
            input_size,
            hidden_size,
            text_decoder_layer_fn,
            num_layers,
            dropout)

        visual_attention = Attention(
            hidden_size, hparams.img_feature_size, hidden_size)

        self.curr_visual_attention = dc(visual_attention)
        self.next_visual_attention = dc(visual_attention)
        self.goal_visual_attention = dc(visual_attention)

        # Intra-task decoder
        state_decoder_layer_fn = lambda input_size, mem_size: StateDecoderLayer(
            input_size, mem_size, hidden_size, attention_heads, dropout, hparams)

        self.state_decoder = Decoder(
            state_size,
            hidden_size,
            state_decoder_layer_fn,
            num_layers,
            dropout)

        # Time features
        self.local_time_embedding  = LearnedTime(hidden_size, hidden_size)
        self.global_time_embedding = LearnedTime(hidden_size * 2, hidden_size)

    def reset(self, batch_size):
        self.text_decoder.reset(batch_size)
        self.state_decoder.reset(batch_size)
        self.last_output = self.text_decoder.value[-1][-1]

        # Reset time
        self.local_time = self.local_time_embedding.reset(batch_size)
        self.global_time = self.global_time_embedding.reset(batch_size)

    def reset_text_decoder(self, batch_size):
        self.text_decoder.reset(batch_size)
        self.last_output = self.text_decoder.value[-1][-1]

        # Reset local time only
        self.local_time = self.local_time_embedding.reset(batch_size)

    def compute_features(self, curr_view_features, goal_view_features):
        scene = self.curr_visual_attention(self.last_output, curr_view_features)
        goal_sim = self.match_view(curr_view_features, goal_view_features)
        goal_sim = goal_sim.unsqueeze(-1).expand(-1, -1, 4).contiguous()\
            .view(-1, goal_sim.size(1) * 4)

        return scene, goal_sim

    def match_view(self, view_a, view_b):
        '''
            Compute a vector representing how similar the current view is to
                the goal view.
        '''
        dot_product = torch.bmm(view_a, view_b.transpose(1, 2))
        view_a_norm = torch.norm(view_a, p=2, dim=2)
        view_b_norm = torch.norm(view_b, p=2, dim=2)
        norm_product = torch.bmm(
            view_a_norm.unsqueeze(2), view_b_norm.unsqueeze(1))
        norm_product = norm_product.clamp(min=1e-8)
        cosine_similarities = dot_product / norm_product
        d, _ = torch.max(cosine_similarities, dim=2)
        return d

    def decode_text(self, input, text_ctx, text_ctx_mask,
            curr_view_features, goal_view_features):
        '''
            Take an inter-task decoding step
        '''

        hidden, output = self.text_decoder(input, self.local_time, text_ctx,
            text_ctx_mask)

        self.last_output = hidden
        self.local_time = self.local_time_embedding()

        c_next = self.next_visual_attention(output, curr_view_features)
        c_goal = self.goal_visual_attention(output, goal_view_features)

        return output, c_next, c_goal

    def decode_state(self, input, mask=None):
        '''
            Take an intra-task decoding step
        '''

        output = self.state_decoder(input, self.global_time, mask=mask)
        self.global_time = self.global_time_embedding(self.local_time)

        return output

    def encode(self, seq, seq_mask):
        return self.encoder(seq, seq_mask)


class EltwiseProdScoring(nn.Module):

    def __init__(self, a_dim, dot_dim):

        super(EltwiseProdScoring, self).__init__()

        self.linear_in_a = nn.Linear(a_dim, dot_dim)
        self.linear_out  = nn.Linear(dot_dim, 1)

    def forward(self, h, all_as):
        all_as = self.linear_in_a(all_as)  # batch x a_num x dot_dim
        eltprod = torch.mul(h.unsqueeze(1), all_as)  # batch x a_num x dot_dim
        logit = self.linear_out(eltprod).squeeze(2)  # batch x a_num
        return logit


class NavModule(Seq2SeqModel):

    def __init__(self, vocab_size, agent_class, hparams, device):

        visual_feature_with_loc_size = \
            hparams.img_feature_size + hparams.loc_embed_size

        hidden_size = hparams.hidden_size

        nav_input_size = visual_feature_with_loc_size + \
            hparams.img_feature_size + hparams.nav_dist_size * 4

        nav_state_size = hidden_size + hparams.img_feature_size * 2 + \
            hparams.nav_dist_size * 4

        super(NavModule, self).__init__(
            vocab_size, nav_input_size, nav_state_size, hparams, device)

        self.start_action = torch.zeros(
            visual_feature_with_loc_size, dtype=torch.float, device=device)

        self.predictor = EltwiseProdScoring(
            visual_feature_with_loc_size, hidden_size)

    def init_action(self, batch_size):
        return self.start_action.expand(batch_size, -1)

    def decode(self, global_time, local_time, prev_a, a_embeds, text_ctx,
            text_ctx_mask, curr_view_features, goal_view_features, logit_mask):

        scene, goal_sim = self.compute_features(
            curr_view_features, goal_view_features)

        text_input = torch.cat((prev_a, scene, goal_sim), dim=-1)
        text_output = self.decode_text(text_input, text_ctx, text_ctx_mask,
            curr_view_features, goal_view_features)

        state_input = torch.cat(text_output + (goal_sim,), dim=-1)
        state_output = self.decode_state(state_input)

        logit = self.predictor(state_output, a_embeds)
        logit.data.masked_fill_(logit_mask, -float('inf'))

        return logit


class AskModule(Seq2SeqModel):

    def __init__(self, vocab_size, agent_class, hparams, device):

        visual_feature_with_loc_size = \
            hparams.img_feature_size + hparams.loc_embed_size

        hidden_size = hparams.hidden_size

        ask_input_size = hparams.ask_embed_size + hparams.img_feature_size + \
            hparams.nav_dist_size * 4

        ask_state_size = hidden_size + hparams.img_feature_size * 2 + \
            hparams.nav_dist_size * 8

        super(AskModule, self).__init__(
            vocab_size, ask_input_size, ask_state_size, hparams, device)

        self.action_embedding = nn.Embedding(agent_class.n_input_ask_actions(),
            hparams.ask_embed_size, agent_class.ask_actions.index('<start>'))

        num_reason_labels = len(AskTeacher.reason_labels)
        self.reason_predictor = nn.Linear(hidden_size, num_reason_labels)

        if hparams.no_reason:
            self.predictor = nn.Linear(
                hidden_size, agent_class.n_output_ask_actions())
        else:
            self.predictor = nn.Linear(
                num_reason_labels, agent_class.n_output_ask_actions())

        self.start_action = torch.ones(1, dtype=torch.long, device=device) * \
            agent_class.ask_actions.index('<start>')

        self.init_action_mask = torch.ones(1, dtype=torch.uint8, device=device)

        self.hparams = hparams

    def reset(self, batch_size):
        super(AskModule, self).reset(batch_size)
        self.action_mask = [self.init_action_mask.expand(batch_size)]

    def update_action_mask(self, mask):
        self.action_mask.append(mask)

    def init_action(self, batch_size):
        return self.start_action.expand(batch_size)

    def decode(self, local_time, global_time, prev_a, nav_dist, text_ctx,
            text_ctx_mask, curr_view_features, goal_view_features, logit_mask):

        scene, goal_sim = self.compute_features(
            curr_view_features, goal_view_features)

        prev_a = self.action_embedding(prev_a)
        text_input = torch.cat((prev_a, scene, goal_sim), dim=-1)
        text_output = self.decode_text(text_input, text_ctx, text_ctx_mask,
            curr_view_features, goal_view_features)

        nav_dist = nav_dist.unsqueeze(-1).expand(-1, -1, 4).contiguous()\
            .view(-1, nav_dist.size(1) * 4)
        state_input = torch.cat(text_output + (goal_sim, nav_dist), dim=-1)

        mask = torch.stack(self.action_mask).transpose(0, 1).contiguous()
        state_output = self.decode_state(state_input, mask=mask)

        if self.hparams.no_reason:
            reason_logit = self.reason_predictor(state_output)
            logit = self.predictor(state_output)
        else:
            reason_logit = self.reason_predictor(state_output)
            logit = self.predictor(reason_logit)

        logit.data.masked_fill_(logit_mask, -float('inf'))

        return logit, reason_logit


class AgentModel(nn.Module):

    def __init__(self, vocab_size, hparams, device):
        super(AgentModel, self).__init__()

        agent_class = AskAgent
        self.nav_module = NavModule(vocab_size, agent_class, hparams, device)
        self.ask_module = AskModule(vocab_size, agent_class, hparams, device)

    def reset(self, batch_size):
        self.nav_module.reset(batch_size)
        self.ask_module.reset(batch_size)

        return self.nav_module.init_action(batch_size), \
               self.ask_module.init_action(batch_size)

    def reset_text_decoder(self, batch_size):
        self.nav_module.reset_text_decoder(batch_size)
        self.ask_module.reset_text_decoder(batch_size)

    def encode(self, *args, **kwargs):
        return self.nav_module.encode(*args, **kwargs), \
               self.ask_module.encode(*args, **kwargs)

    def decode_nav(self, *args, **kwargs):
        return self.nav_module.decode(*args, **kwargs)

    def decode_ask(self, *args, **kwargs):
        return self.ask_module.decode(*args, **kwargs)



