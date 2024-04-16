from io import open
import re
from collections import Counter
import torch
import torch.nn as nn
import jieba
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import pickle

from model_attention import Seq2SeqAttention
from model_transformer import Seq2SeqTransformer
from training import model_training, feedforward
from visualization import print_sentences

# Redirect output to os.devnull
jieba.default_logger.setLevel(20)


# supports MacOS mps and CUDA
def GPU_Device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'


# bidirectional dictionary
class BidirectionalMap:
    def __init__(self):
        self.key_to_value = {}
        self.value_to_key = {}
        self.add_mapping('<pad>', 0)
        self.add_mapping('<sos>', 1)
        self.add_mapping('<eos>', 2)
        self.add_mapping('<unk>', 3)
    
    def __len__(self):
        return len(self.key_to_value)
    
    def keys(self):
        return self.key_to_value.keys()
    
    def add_mapping(self, key, value):
        self.key_to_value[key] = value
        self.value_to_key[value] = key

    def get_value(self, key):
        return self.key_to_value.get(key, 3)

    def get_key(self, value):
        return self.value_to_key.get(value, '<unk>')
    
    
# model inference
def inference(model, X, target_vocab, device, max_length):
    with torch.no_grad():
        # Set the encoder to evaluation mode
        model.eval()

        # Move input tensor to device
        X = X.to(device)
        
        # inference
        out = model.inference(X)
        
        decoded_words = tensorToTokens(target_vocab, out)
        return decoded_words
    

def en_tokenizer(text):
    # Define regex pattern to match words
    pattern = r"\w+|[^\w\s]"
    tokens = re.findall(pattern, text)
    tokens = [token.lower() for token in tokens] # convert to lower case
    return tokens


# tokenize each sentence
def tokenize(sentences, tokenizer):
    return [list(tokenizer(sentence.strip())) for sentence in tqdm(sentences)]


# build word to ind dictionary
def build_vocab(sentences, min_freq=2):
    # Initialize an empty Counter object to count word frequencies
    word_counts = Counter()
    
    # Count word frequencies in all sentences
    for tokens in sentences:
        word_counts.update(tokens)
    
    # Create a vocabulary mapping from words to indices
    vocab = BidirectionalMap()
    
    for word, freq in word_counts.items():
        # ignore the word count with too few frequency
        if freq < min_freq: 
            continue
        if word not in vocab.keys():
            vocab.add_mapping(word, len(vocab))
           
    return vocab
            

# remove the sentence pair that contains unk (too few occurences)
def remove_unk(en_sentences, zh_sentences, en_vocab, zh_vocab):
    removed_count = 0
    i = 0

    while i < len(en_sentences):
        en_tokens = en_sentences[i]
        zh_tokens = zh_sentences[i]

        # Check if any token in the English sentence is unknown
        if any(token not in en_vocab for token in en_tokens) or \
           any(token not in zh_vocab for token in zh_tokens):
            del en_sentences[i]
            del zh_sentences[i]
            removed_count += 1
        else:
            i += 1

    return removed_count



# convert from a sentence to a torch tensor
def tokensToTensor(lang, sentence, max_length):
    # Tokenize sentence and convert tokens to indices using the vocabulary
    tokens = [lang.get_value(token) for token in sentence]
    tokens = tokens[:max_length - 1]  # Truncate sentence if it's longer than max_length - 1
    
    # tokens input has <sos> in the front
    tokens_input = tokens.copy()
    tokens_input.insert(0, lang.get_value('<sos>'))
    tokens_input += [lang.get_value('<pad>')] * (max_length - len(tokens_input))
    tokens_input = torch.tensor(tokens_input, dtype=torch.long)
    
    # tokens output has <eos> at the end
    tokens_output = tokens.copy()
    tokens_output.append(lang.get_value('<eos>'))  # Append end-of-sequence token
    tokens_output += [lang.get_value('<pad>')] * (max_length - len(tokens_output))
    tokens_output = torch.tensor(tokens_output, dtype=torch.long)

    return tokens_input, tokens_output
    


# convert from a torch tensor to sentence
def tensorToTokens(lang, tensor):
    # Convert tensor to list
    tokens = tensor.tolist()
    
    # replace the last ind with eos (incase eos is not in sentence)
    tokens[-1] = lang.get_value('<eos>')
    
    # Find the index of the end-of-sequence token
    eos_index = tokens.index(lang.get_value('<eos>'))
    # Remove padding tokens and end-of-sequence token
    tokens = tokens[:eos_index]
    
    # Convert indices back to tokens
    tokens = [lang.get_key(token) for token in tokens]
    return tokens


def dataloader(source_sentences, target_sentences_input, target_sentences_output, batch_size, model_name):    
    # for Attention models, target_sentences_input should be replaced by target_sentences_output
    if model_name == 'Attention':
        target_sentences_input = target_sentences_output
        
    # Combine source and target sentences into tuples
    dataset = list(zip(source_sentences, target_sentences_input, target_sentences_output))
    
    # Create DataLoader with zipped dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

    
    
def main():
    # which model to use
    model_name = 'Transformer'
    
    # Define paths to the extracted files
    en_file = '../Datasets/Machine Translation Between Chinese and English/english.en'
    zh_file = '../Datasets/Machine Translation Between Chinese and English/chinese.zh'
    
    # Read English sentences from the file
    with open(en_file, 'r', encoding='utf-8') as f:
        en_sentences = f.readlines()[:100000] # load a small data subset
    # Read Chinese sentences from the file
    with open(zh_file, 'r', encoding='utf-8') as f:
        zh_sentences = f.readlines()[:100000] # load a small data subset
    print('reading files completed')
    
    # visualize examples
    for i in range(0, 500, 100):
        print_sentences(en_sentences[i], zh_sentences[i])
        
    # tokenize
    en_sentences = tokenize(en_sentences, en_tokenizer)
    zh_sentences = tokenize(zh_sentences, jieba.cut)
    
    # building the vocabulary
    if 'English.pkl' in os.listdir():
        with open('English.pkl', 'rb') as f:    
            en_vocab = pickle.load(f)
    else:
        en_vocab = build_vocab(en_sentences)
        with open('English.pkl', 'wb') as f:
            pickle.dump(en_vocab, f)
            
    if 'Chinese.pkl' in os.listdir():
        with open('Chinese.pkl', 'rb') as f:    
            zh_vocab = pickle.load(f)
    else:
        zh_vocab = build_vocab(zh_sentences)
        with open('Chinese.pkl', 'wb') as f:
            pickle.dump(zh_vocab, f)
    print('done building the vocabulary')
    
    
    # remove the sentences pair that contains unk (not in vocab)
    num = remove_unk(en_sentences, zh_sentences, en_vocab.keys(), zh_vocab.keys())
    print('number of sentences removed', num)
    
    # train, valid, and test split
    trainX, trainY = en_sentences[:int(0.8*len(en_sentences))], zh_sentences[:int(0.8*len(zh_sentences))]
    validX, validY = en_sentences[int(0.8*len(en_sentences)):int(0.9*len(en_sentences))], zh_sentences[int(0.8*len(zh_sentences)):int(0.9*len(zh_sentences))]
    testX, testY = en_sentences[int(0.9*len(en_sentences)):], zh_sentences[int(0.9*len(zh_sentences)):]
    print('done train test split')
    
    # convert train and valid to tensor
    max_len = 64
    trainX = [tokensToTensor(en_vocab, item, max_len)[1] for item in trainX]
    trainY_input = [tokensToTensor(zh_vocab, item, max_len)[0] for item in trainY]
    trainY_output = [tokensToTensor(zh_vocab, item, max_len)[1] for item in trainY]
    
    validX = [tokensToTensor(en_vocab, item, max_len)[1] for item in validX]
    validY_input = [tokensToTensor(zh_vocab, item, max_len)[0] for item in validY]
    validY_output = [tokensToTensor(zh_vocab, item, max_len)[1] for item in validY]
    print('done convert to tensor')
    
    # create train and valid data loader
    batch_size = 100 # 128 is too large for transformer
    train_dataloader = dataloader(trainX, trainY_input, trainY_output, batch_size, model_name)
    valid_dataloader = dataloader(validX, validY_input, validY_output, batch_size, model_name)
    print('done created loader')
    
    # define / load the model
    src_pad_ind = en_vocab.get_value('<pad>')
    trg_pad_ind = zh_vocab.get_value('<pad>')
    trg_sos_ind = zh_vocab.get_value('<sos>')
    trg_eos_ind = zh_vocab.get_value('<eos>')
    
    if model_name == 'Attention':
        model = Seq2SeqAttention(len(en_vocab), len(zh_vocab), max_len, trg_sos_ind)
    elif model_name == 'Transformer':
        model = Seq2SeqTransformer(len(en_vocab), len(zh_vocab), src_pad_ind, trg_pad_ind, \
                                   trg_sos_ind, trg_eos_ind, max_len, num_layers=3)
    model = model.to(GPU_Device()) # move the model to GPU
    if f'{type(model).__name__}.pth' in os.listdir():
        model.load_state_dict(torch.load(f'{type(model).__name__}.pth'))
    
    # model training
    model_training(train_dataloader, valid_dataloader, model, GPU_Device())
    
    # load the best model
    model.load_state_dict(torch.load(f'{type(model).__name__}.pth'))
    
    # load the test dataset
    testX = [tokensToTensor(en_vocab, item, max_len)[1] for item in testX]
    testY_input = [tokensToTensor(zh_vocab, item, max_len)[0] for item in testY]
    testY_output = [tokensToTensor(zh_vocab, item, max_len)[1] for item in testY]
    test_dataloader = dataloader(testX, testY_input, testY_output, batch_size, model_name)
    criterion = nn.NLLLoss()
    test_loss, test_blue = feedforward(test_dataloader, model, criterion, GPU_Device())
    print(f'Test BLUE: {test_blue:.3f} | Test Loss: {test_loss:.3f}')
    
    for ind in range(0, 500, 10):
        sentenceX = ' '.join(tensorToTokens(en_vocab, testX[ind]))
        sentenceY = ''.join(tensorToTokens(zh_vocab, testY_output[ind]))
        predY = inference(model, testX[ind].unsqueeze(0), zh_vocab, GPU_Device(), max_len)
        sentencePred = ''.join(predY)
        print_sentences(sentenceX, sentenceY, sentencePred)


if __name__ == "__main__":
    main()
