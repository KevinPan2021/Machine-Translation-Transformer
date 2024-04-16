import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# For every input word the encoder outputs a vector and a hidden state, 
# and uses the hidden state for the next input word.
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, input_tensor):
        embedded = self.dropout(self.embedding(input_tensor))
        output, hidden = self.gru(embedded)
        return output, hidden
    
    
    

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights
    
    
        
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, MaxLen, dropout_p, SOS_token):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        
        self.SOS_token = SOS_token
        self.MaxLen = MaxLen
        
    
    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
    
    
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        device = encoder_outputs.device  # Get the device of encoder_outputs
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(self.SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(self.MaxLen):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions



# seqence to sequence model with encoder/decoder
class Seq2SeqAttention(nn.Module):
    def __init__(self, input_size, output_size, maxlen, trg_sos_token, hidden_size=128, dropout=0.1):
        super(Seq2SeqAttention, self).__init__()

        self.encoder = EncoderRNN(input_size, hidden_size, dropout)
        self.decoder = AttnDecoderRNN(hidden_size, output_size, maxlen, dropout, trg_sos_token)

    
    def forward(self, input_tensor, target_tensor=None):
        # for torchsummary (only inputs float data type)
        input_tensor = input_tensor.to(dtype=torch.long) 
        if not target_tensor is None:
            target_tensor = target_tensor.to(dtype=torch.long)
        
        encoder_outputs, encoder_hidden = self.encoder(input_tensor)
        decoder_outputs, _, attentions = self.decoder(encoder_outputs, encoder_hidden, target_tensor)
        return decoder_outputs
    
    
    def inference(self, X):
        decoder_outputs = self.forward(X)
        
        # greedy search
        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()
        
        return decoded_ids # remove the sos token
            
            
            
def main():
    seq_len = 64
    trg_sos_token = 1
    src_vocab_size = 1000
    trg_vocab_size = 1000
    model = Seq2SeqAttention(src_vocab_size, trg_vocab_size, seq_len, trg_sos_token)
    summary(model, [(seq_len,), (seq_len,)])
    
    
if __name__ == "__main__":
    main()
