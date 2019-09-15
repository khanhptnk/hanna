""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn
from termcolor import colored

class Attention(nn.Module):

    def __init__(self, h_dim, v_dim, dot_dim):

        super(Attention, self).__init__()

        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_v = nn.Linear(v_dim, dot_dim, bias=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, query, values, mask=None, print_prob=None):
        target = self.linear_in_h(query).unsqueeze(2)  # batch x dot_dim x 1
        context = self.linear_in_v(values)  # batch x v_num x dot_dim

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x v_num
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)

        if print_prob is not None:
            top_attn = attn[print_prob].tolist()
            max_elm = max(top_attn)
            for i, x in enumerate(top_attn):
                rep = f'( {i} {(x * 100):.2f} )'
                if abs(x - max_elm) < 1e-6:
                    rep = colored(rep, 'red')
                print(rep, end=' ')
            print()

        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x v_num

        weighted_context = torch.bmm(attn3, values).squeeze(1)  # batch x v_dim

        return weighted_context


class SimAttention(nn.Module):

    def __init__(self):

        super(SimAttention, self).__init__()

    def forward(self, query, key, value, mask=None, print_prob=None):

        dot_product  = torch.bmm(key, query.unsqueeze(2)).squeeze(2) # batch x v_num
        norm_query   = torch.norm(query, p=2, dim=1) # batch
        norm_key     = torch.norm(key, p=2, dim=2) # batch x v_num
        norm_product = norm_query.unsqueeze(1) * norm_key # batch x v_num
        norm_product = norm_product.clamp(min=1e-8)

        attn = dot_product / norm_product

        if print_prob is not None:
            print(mask[print_prob].tolist())
            top_attn = attn[print_prob].tolist()
            max_elm = max(top_attn)
            for i, x in enumerate(top_attn):
                rep = f'( {i} {x:.2f} )'
                if abs(x - max_elm) < 1e-6:
                    rep = colored(rep, 'red')
                print(rep, end=' ')
            print()


        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, 0)

        attn[attn < 0.9] = 0
        #attn = attn / key.size(1)

        #attn = (attn + 1) / (2 * key.size(1))
        attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)

        weighted_context = torch.bmm(attn.unsqueeze(1), value).squeeze(1)

        return weighted_context


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, query_dim, key_dim, value_dim, model_dim, head_count):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_key   = nn.Linear(key_dim, model_dim)
        self.linear_value = nn.Linear(value_dim, model_dim)
        self.linear_query = nn.Linear(query_dim, model_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value, mask=None, print_prob=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask indicating which keys have
               non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        query_ndim = query.dim()
        if query_ndim == 2:
            query = query.unsqueeze(1)

        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(1)

        # 1) Project key, value, and query.
        key = shape(self.linear_key(key))
        value = shape(self.linear_value(value))
        query = shape(self.linear_query(query))

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        scores = query_key
        scores = scores.float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)

        context_original = torch.matmul(attn, value)

        context = unshape(context_original)

        output = self.final_linear(context)

        # Return one attn
        if print_prob is not None:
            print(attn.size())
            for i in range(self.head_count):
                top_attn = attn \
                    .view(batch_size, head_count,
                          query_len, key_len)[:, i, :, :] \
                    .contiguous()
                top_attn = top_attn[print_prob].tolist()[0]
                max_elm = max(top_attn)
                for i, x in enumerate(top_attn):
                    rep = f'( {i} {(x * 100):.2f} )'
                    if abs(x - max_elm) < 1e-6:
                        rep = colored(rep, 'green')
                    print(rep, end=' ')
                print()

        if query_ndim == 2:
            output = output.squeeze(1)

        return output
