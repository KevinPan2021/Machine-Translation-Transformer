from io import open
import torch
from torch.utils.data import Dataset, DataLoader
import os

from tokenizer import Tokenizer
from transformer import Transformer
from training import model_training, feedforward
from visualization import print_sentences


# supports MacOS mps and CUDA
def compute_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
    


# translation dataset inherent from Dataset class
class Translation_Dataset(Dataset):
    def __init__(self, X, Y, seq_length, tokenizer):
        self.X = X
        self.Y = Y
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        
        
    def __getitem__(self, idx):
        # tokenize
        x = self.tokenizer.encode(self.X[idx])
        y = self.tokenizer.encode(self.Y[idx])
        special_tok = self.tokenizer.get_special_token()
        
        # add <|endoftext|> token to the end of x
        x = x[:self.seq_length-1] # final x shouldn't exceed seq_length
        x.append(special_tok['<|endoftext|>'])
        
        # add <|startoftext|> token to the beginning of y_in
        y_in = y[:self.seq_length-1] # final y shouldn't exceed seq_length
        y_in.insert(0, special_tok['<|startoftext|>'])
        
        # add <|endoftext|> token to the end of y_out
        y_out = y[:self.seq_length-1] # final y shouldn't exceed seq_length
        y_out.append(special_tok['<|endoftext|>'])
        
        # Pad x and y to seq_length
        x = x + [special_tok['<|pad|>']] * (self.seq_length - len(x))
        y_in = y_in + [special_tok['<|pad|>']] * (self.seq_length - len(y_in))
        y_out = y_out + [special_tok['<|pad|>']] * (self.seq_length - len(y_out))
        
        # Create y_attention_mask where actual tokens are 1 and padding tokens are 0
        y_mask = [1] * len(y_in) + [0] * (self.seq_length - len(y_in))
        
        return {
            'x': torch.tensor(x, dtype=torch.long),
            'y_attention_mask': torch.tensor(y_mask, dtype=torch.long),
            
            'y_in': torch.tensor(y_in, dtype=torch.long),
            'y_out': torch.tensor(y_out, dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.Y)
    
    
    

def main():
    # Define paths to the extracted files
    en_file = '../Datasets/Machine Translation Between Chinese and English/english.en'
    zh_file = '../Datasets/Machine Translation Between Chinese and English/chinese.zh'
    
    # Read English sentences from the file
    with open(en_file, 'r', encoding='utf-8') as f:
        en_sentences = f.readlines()
    # Read Chinese sentences from the file
    with open(zh_file, 'r', encoding='utf-8') as f:
        zh_sentences = f.readlines()
    print('reading files completed')
    
    # visualize examples
    for i in range(0, len(en_sentences), len(en_sentences)//6):
        print_sentences(en_sentences[i], zh_sentences[i])
        
    # tokenizer
    tokenizer = Tokenizer()
    vocab_path = 'vocab.pkl'
    # if a tokenizer has been trained and saved
    if os.path.exists(vocab_path): 
        tokenizer.load(vocab_path)
    else:
        # combine the two languages
        lang_strings = en_sentences + zh_sentences
        tokenizer.train(lang_strings, iteration=10000)
        # save the tokenizer
        tokenizer.save(vocab_path)
    print('done loading tokenizer')
    
    # train, valid, and test split
    trainX, trainY = en_sentences[:int(0.8*len(en_sentences))], zh_sentences[:int(0.8*len(zh_sentences))]
    validX, validY = en_sentences[int(0.8*len(en_sentences)):int(0.9*len(en_sentences))], zh_sentences[int(0.8*len(zh_sentences)):int(0.9*len(zh_sentences))]
    testX, testY = en_sentences[int(0.9*len(en_sentences)):], zh_sentences[int(0.9*len(zh_sentences)):]
    print('done train test split')
    
    # convert train and valid to tensor
    seq_len = 64
    train_dataset = Translation_Dataset(trainX, trainY, seq_len, tokenizer)
    valid_dataset = Translation_Dataset(validX, validY, seq_len, tokenizer)
    
    # create train and valid data loader
    train_loader = DataLoader(
        train_dataset, batch_size=64, num_workers=4, pin_memory=True, 
        persistent_workers=True, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=128, num_workers=4, pin_memory=True, 
        persistent_workers=True, shuffle=False
    )
    print('done created loader')
    
    
    # visualize the tokenized result
    for i in range(0, len(train_dataset), len(train_dataset)//6):
        item = train_dataset[i]
        X = tokenizer.decode(item['x'].numpy())
        Y_in = tokenizer.decode(item['y_in'].numpy())
        Y_out = tokenizer.decode(item['y_out'].numpy())
        print(X.strip())
        print(Y_in.strip())
        print(Y_out.strip())
        print()
    
    
    # model definition
    pad_ind = tokenizer.get_special_token()['<|pad|>']
    start_ind = tokenizer.get_special_token()['<|startoftext|>']
    end_ind = tokenizer.get_special_token()['<|endoftext|>']
    
    d_model = 640
    num_layers = 8
    num_heads = 10
    d_ff = 2048
    model = Transformer(
        tokenizer.vocab_size(), 
        pad_ind, start_ind, end_ind,
        seq_len, num_heads=num_heads, num_layers=num_layers,
        d_model=d_model, d_ff=d_ff
    )
    model = model.to(compute_device()) # move the model to GPU
    
    # model training
    model_training(train_loader, valid_loader, model)
    
    # load the best model
    model.load_state_dict(torch.load(f'{type(model).__name__}.pth'))
    
    # load the test dataset
    test_dataset = Translation_Dataset(testX, testY, seq_len, tokenizer)
    
    # create train and valid data loader
    test_loader = DataLoader(
        test_dataset, batch_size=128, num_workers=4, pin_memory=True, 
        persistent_workers=True, shuffle=False
    )
    
    test_loss, test_blue = feedforward(test_loader, model)
    print(f'Test BLUE: {test_blue:.3f} | Test Loss: {test_loss:.3f}')
    
    # visualize some outputs 
    for i in range(0, len(test_dataset), len(test_dataset)//6):
        item = test_dataset[i]
        
        # delete all special tokens
        sentenceX = tokenizer.decode(item['x'].numpy(), omit_special_tok=True)
        sentenceY = tokenizer.decode(item['y_in'].numpy(), omit_special_tok=True)
        
        # conver to tensor
        x = item['x'].unsqueeze(0).to(compute_device())
        
        # model inference
        out_token = model.inference(x).to('cpu')
        
        # token decode
        sentencePred = tokenizer.decode(out_token.numpy(), omit_special_tok=True)
        
        print_sentences(sentenceX, sentenceY, sentencePred)


if __name__ == "__main__":
    main()
