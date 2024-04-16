import torch
import torch.nn as nn
import math
from torchsummary import summary



class MultiheadAttention(nn.Module):
    def __init__(self, d_model, dropout, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // self.num_heads
        
        self.inf = 1e9

        # W^Q, W^K, W^V in the paper
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attn_softmax = nn.Softmax(dim=-1)

        # Final output linear transformation
        self.w_0 = nn.Linear(d_model, d_model)


    def forward(self, q, k, v, mask=None):
        input_shape = q.shape

        # Linear calculation +  split into num_heads
        q = self.w_q(q).view(input_shape[0], -1, self.num_heads, self.d_k) # (B, L, num_heads, d_k)
        k = self.w_k(k).view(input_shape[0], -1, self.num_heads, self.d_k) # (B, L, num_heads, d_k)
        v = self.w_v(v).view(input_shape[0], -1, self.num_heads, self.d_k) # (B, L, num_heads, d_k)

        # For convenience, convert all tensors in size (B, num_heads, L, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Conduct self-attention
        attn_values = self.self_attention(q, k, v, mask=mask) # (B, num_heads, L, d_k)
        concat_output = attn_values.transpose(1, 2)\
            .contiguous().view(input_shape[0], -1, self.d_model) # (B, L, d_model)

        return self.w_0(concat_output)


    def self_attention(self, q, k, v, mask=None):
        # Calculate attention scores with scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) # (B, num_heads, L, L)
        attn_scores = attn_scores / math.sqrt(self.d_k)

        # If there is a mask, make masked spots -INF
        if mask is not None:
            mask = mask.unsqueeze(1) # (B, 1, L) => (B, 1, 1, L) or (B, L, L) => (B, 1, L, L)
            attn_scores = attn_scores.masked_fill_(mask == 0, -1 * self.inf)

        # Softmax and multiplying K to calculate attention value
        attn_distribs = self.attn_softmax(attn_scores)

        attn_distribs = self.dropout(attn_distribs)
        attn_values = torch.matmul(attn_distribs, v) # (B, num_heads, L, d_k)

        return attn_values
    
    

class FeedFowardLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.linear_1(x)) # (B, L, d_ff)
        x = self.dropout(x)
        x = self.linear_2(x) # (B, L, d_model)

        return x


class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.layer = nn.LayerNorm([d_model], elementwise_affine=True, eps=self.eps)

    def forward(self, x):
        x = self.layer(x)

        return x



class PositionalEncoder(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        
        self.d_model = d_model
        
        # Make initial positional encoding matrix with 0
        pe_matrix= torch.zeros(seq_len, d_model) # (L, d_model)

        # Calculating position encoding values
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / self.d_model)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / self.d_model)))

        pe_matrix = pe_matrix.unsqueeze(0) # (1, L, d_model)
        self.positional_encoding = pe_matrix.requires_grad_(False)

    def forward(self, x):
        x = x * math.sqrt(self.d_model) # (B, L, d_model)
        x = x + self.positional_encoding.to(x.device) # (B, L, d_model)

        return x




class EncoderLayer(nn.Module):
    def __init__(self, dropout, d_model, num_heads, d_ff):
        super().__init__()
        self.layer_norm_1 = LayerNormalization(d_model)
        self.multihead_attention = MultiheadAttention(d_model, dropout, num_heads)
        self.drop_out_1 = nn.Dropout(dropout)

        self.layer_norm_2 = LayerNormalization(d_model)
        self.feed_forward = FeedFowardLayer(d_model, d_ff, dropout)
        self.drop_out_2 = nn.Dropout(dropout)

    def forward(self, x, e_mask):
        x_1 = self.layer_norm_1(x) # (B, L, d_model)
        x = x + self.drop_out_1(
            self.multihead_attention(x_1, x_1, x_1, mask=e_mask)
        ) # (B, L, d_model)
        x_2 = self.layer_norm_2(x) # (B, L, d_model)
        x = x + self.drop_out_2(self.feed_forward(x_2)) # (B, L, d_model)

        return x # (B, L, d_model)


class DecoderLayer(nn.Module):
    def __init__(self, dropout, d_model, num_heads, d_ff):
        super().__init__()
        self.layer_norm_1 = LayerNormalization(d_model)
        self.masked_multihead_attention = MultiheadAttention(d_model, dropout, num_heads)
        self.drop_out_1 = nn.Dropout(dropout)

        self.layer_norm_2 = LayerNormalization(d_model)
        self.multihead_attention = MultiheadAttention(d_model, dropout, num_heads)
        self.drop_out_2 = nn.Dropout(dropout)

        self.layer_norm_3 = LayerNormalization(d_model)
        self.feed_forward = FeedFowardLayer(d_model, d_ff, dropout)
        self.drop_out_3 = nn.Dropout(dropout)

    def forward(self, x, e_output, e_mask,  d_mask):
        x_1 = self.layer_norm_1(x) # (B, L, d_model)
        x = x + self.drop_out_1(
            self.masked_multihead_attention(x_1, x_1, x_1, mask=d_mask)
        ) # (B, L, d_model)
        x_2 = self.layer_norm_2(x) # (B, L, d_model)
        x = x + self.drop_out_2(
            self.multihead_attention(x_2, e_output, e_output, mask=e_mask)
        ) # (B, L, d_model)
        x_3 = self.layer_norm_3(x) # (B, L, d_model)
        x = x + self.drop_out_3(self.feed_forward(x_3)) # (B, L, d_model)

        return x # (B, L, d_model)



        
class Encoder(nn.Module):
    def __init__(self, num_layers, dropout, d_model, num_heads, d_ff):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([EncoderLayer(dropout, d_model, num_heads, d_ff) for i in range(self.num_layers)])
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, e_mask):
        for i in range(self.num_layers):
            x = self.layers[i](x, e_mask)

        return self.layer_norm(x)


class Decoder(nn.Module):
    def __init__(self, num_layers, dropout, d_model, num_heads, d_ff):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([DecoderLayer(dropout, d_model, num_heads, d_ff) for i in range(self.num_layers)])
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, e_output, e_mask, d_mask):
        for i in range(self.num_layers):
            x = self.layers[i](x, e_output, e_mask, d_mask)

        return self.layer_norm(x)
    


# main transformer class
class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_ind, trg_pad_ind, trg_sos_ind, 
                 trg_eos_ind, seq_len, num_heads=8, num_layers=6, d_model=512, 
                 d_ff=2048, dropout=0.1):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_pad_ind = src_pad_ind
        self.trg_pad_ind = trg_pad_ind
        self.trg_sos_ind = trg_sos_ind
        self.trg_eos_ind = trg_eos_ind
        self.seq_len = seq_len
        
        self.num_heads = num_heads # number of multi-head
        self.num_layers = num_layers # number of encoders / decoders
        self.d_model = d_model # position encoding input size
        self.d_ff = d_ff # forward expansion
        self.dropout = dropout # dropout rate
        
        self.src_embedding = nn.Embedding(self.src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(self.trg_vocab_size, d_model)
        self.positional_encoder = PositionalEncoder(self.seq_len, self.d_model)
        
        self.encoder = Encoder(self.num_layers, self.dropout, self.d_model, self.num_heads, self.d_ff)
        self.decoder = Decoder(self.num_layers, self.dropout, self.d_model, self.num_heads, self.d_ff)
        self.output_linear = nn.Linear(d_model, self.trg_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    
    # mask padding index
    def generate_mask(self, src_input):
        return (src_input != self.src_pad_ind).unsqueeze(1)  # (B, 1, L)
    
    
    # mask padding index and create square subsequent mask
    def generate_tril_mask(self, trg_input):
        d_mask = (trg_input != self.trg_pad_ind).unsqueeze(1)  # (B, 1, L)

        nopeak_mask = torch.ones([1, self.seq_len, self.seq_len], dtype=torch.bool)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
        
        # put into the same device as trg_input
        nopeak_mask = nopeak_mask.to(trg_input.device)
        
        return d_mask & nopeak_mask  # (B, L, L) padding false
        
        
        
    def forward(self, src_input, trg_input):
        # for torchsummary (only inputs float data type)
        src_input = src_input.to(dtype=torch.long)
        trg_input = trg_input.to(dtype=torch.long)
        
        # create masks
        e_mask = self.generate_mask(src_input)
        d_mask = self.generate_tril_mask(trg_input)
            
        # words and position embedding
        src_input = self.src_embedding(src_input).to(src_input.device) # (B, L) => (B, L, d_model)
        trg_input = self.trg_embedding(trg_input).to(trg_input.device) # (B, L) => (B, L, d_model)
        src_input = self.positional_encoder(src_input) # (B, L, d_model) => (B, L, d_model)
        trg_input = self.positional_encoder(trg_input) # (B, L, d_model) => (B, L, d_model)

        e_output = self.encoder(src_input, e_mask) # (B, L, d_model)
        d_output = self.decoder(trg_input, e_output, e_mask, d_mask) # (B, L, d_model)

        output = self.softmax(self.output_linear(d_output)) # (B, L, d_model) => # (B, L, trg_vocab_size)
    
        return output
    
    
    
    def inference(self, src_input):
        # construct trg_input
        trg_input = torch.LongTensor([self.trg_pad_ind] * self.seq_len).to(src_input.device) # (L)
        trg_input[0] = self.trg_sos_ind # assign sos token
        
        # create masks
        e_mask = self.generate_mask(src_input)

        # words and position embedding
        src_input = self.src_embedding(src_input).to(src_input.device) # (B, L) => (B, L, d_model)
        src_input = self.positional_encoder(src_input) # (B, L, d_model) => (B, L, d_model)
        
        e_output = self.encoder(src_input, e_mask) # (B, L, d_model)
        
        cur_len = 1
        for i in range(self.seq_len):
            d_mask = self.generate_tril_mask(trg_input)
            
            trg_embedded = self.trg_embedding(trg_input).to(trg_input.device) # (B, L) => (B, L, d_model)
            trg_encoded = self.positional_encoder(trg_embedded) # (B, L, d_model) => (B, L, d_model)
            
            d_output = self.decoder(trg_encoded, e_output,  e_mask, d_mask) # (1, L, d_model)
            
            output = self.softmax(self.output_linear(d_output) ) # (1, L, trg_vocab_size)
 
            # greedy search
            decoder_outputs = torch.argmax(output, dim=-1) # (1, L)
            
            last_word = decoder_outputs[0][i].item()
            
            if i < self.seq_len-1:
                trg_input[i+1] = last_word
                cur_len += 1
                
            # generated <eos> token
            if last_word == self.trg_eos_ind:
                break
        
        return trg_input[1:]
    

def main():
    seq_len = 64
    src_pad_ind = 0
    trg_pad_ind = 0
    trg_sos_ind = 1
    trg_eos_ind = 2
    src_vocab_size = 1000
    trg_vocab_size = 1000
    num_layers = 3
    model = Seq2SeqTransformer(src_vocab_size, trg_vocab_size, src_pad_ind, trg_pad_ind, trg_sos_ind, trg_eos_ind, seq_len, num_layers=num_layers)
    summary(model, [(seq_len,), (seq_len,)])
    
    
    
if __name__ == "__main__":
    main()
