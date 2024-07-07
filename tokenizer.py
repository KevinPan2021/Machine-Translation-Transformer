# tiktoken (openAI) without forcing splits
import regex as re
from tqdm import tqdm
import pickle
from collections import Counter

class Tokenizer():
    
    def __init__(self, ):
        self._merges = {}
        self._vocab = {idx: bytes([idx]) for idx in range(256)}
        self._special_tok = ['<|endoftext|>', '<|startoftext|>', '<|pad|>']
    

    # count the number of paired occurences
    # input is string of UTF-8 encoded ids, return a dict {(ids1, ids2): occurences}
    def _get_stats(self, ids):
        return Counter(zip(ids, ids[1:]))


    # in the list of ints (ids), replace all occurences of pair with the idx
    def _merge(self, ids, pair, idx):
        pair_len = len(ids) - 1
        i = 0
        new_ids = []
        
        while i < pair_len:
            if ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
    
        # Append the last element if it was not part of the pair
        if i == pair_len:
            new_ids.append(ids[-1])
        
        return new_ids

    
    # byte pair encoding
    def _BPE(self, lang_strings, iteration):
        print('Starting BPE algorithm')
        # concatenate lang_strings into a single long string
        text = ''.join(lang_strings)
        
        # use UTF-8 encoding 
        tokens = text.encode('utf-8') # convert string into raw bytes
        tokens = list(map(int, tokens)) # convert to int [0, 255]
        
        # make a copy
        ids = list(tokens)
        
        # number of iterations for BPE
        with tqdm(total=iteration) as pbar:
            for i in range(iteration):
                # get the pairing stats
                stats = self._get_stats(ids)
                
                # extract the most frequent pair
                top_pair = max(stats, key=stats.get)
                
                idx = 256 + i
                ids = self._merge(ids, top_pair, idx)
                
                self._merges[top_pair] = idx
                
                # Update tqdm description with loss and BLUE
                pbar.set_postfix({
                    'Vocab': idx, 
                    'IdsLen': len(ids),
                    'Ratio': round(len(tokens)/len(ids), 3)
                })
                pbar.update(1)
                
    
    # building vocab from merges
    def _build_vocab(self):
        for (p0, p1), idx in self._merges.items():
            self._vocab[idx] = self._vocab[p0] + self._vocab[p1]
    
    
    # get the size of vocab
    def vocab_size(self):
        return len(self._vocab) + len(self._special_tok)
    
    
    # training the tokenizer using BPE
    def train(self, lang_strings, iteration):
        # calculating the merges using BPE
        self._BPE(lang_strings, iteration)
        # building the new vocab based on merges
        self._build_vocab()
        
        
    # save the merges and vocab dict to file
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self._merges, f)
            pickle.dump(self._vocab, f)
            
        
    # load the merges and vocab dict from file
    def load(self, path):
        with open(path, 'rb') as f:    
            self._merges = pickle.load(f)
            self._vocab = pickle.load(f)
        
        
    # add more special tokens
    def add_special_token(self, token_string):
        # check if the special is already present
        if token_string in self._special_tok:
            print(f'The token {token_string} already exists, skip')
            return
        
        self._special_tok.append(token_string)
        
        
    # get special token
    def get_special_token(self):
        # Create the special token map
        return {
            self._special_tok[i]: len(self._vocab) + i for i in range(len(self._special_tok))
        }
        
        
        
    # convert from ids to text
    def decode(self, ids, omit_special_tok=False):
        vocab = self._vocab.copy()
        
        # Add special tokens
        for i in range(len(self._special_tok)):
            vocab[len(vocab)] = self._special_tok[i]
        
        # omiting the special tokens
        if omit_special_tok:
            tokens = b''.join(b'' if isinstance(vocab[idx], str) else vocab[idx] for idx in ids)
        else:
            # Create a byte string from the list of IDs
            tokens = b''.join(vocab[idx].encode('utf-8') if isinstance(vocab[idx], str) else vocab[idx] for idx in ids)

        # Decode the byte string to text
        text = tokens.decode('utf-8', errors='replace')
        
        return text

    
    
    # convert from text to ids
    def encode(self, text):
        # Create the special token map
        special_tok_map = self.get_special_token()
        
        # Regular expression pattern to match special tokens
        special_tokens_pattern = re.compile('|'.join(re.escape(token) for token in self._special_tok))
        
        # Split the text by special tokens and keep them in the result
        parts = special_tokens_pattern.split(text)
        special_tokens = special_tokens_pattern.findall(text)
        
        tokens = []
        
        # Process each part of the text
        for part in parts:
            # Encode non-special token text to a list of UTF-8 byte values
            if part:
                part_tokens = list(part.encode('utf-8'))
                while len(part_tokens) >= 2:
                    stats = self._get_stats(part_tokens)
                    pair = min(stats, key=lambda p: self._merges.get(p, float('inf')))
                    if pair not in self._merges:
                        break
                    idx = self._merges[pair]
                    part_tokens = self._merge(part_tokens, pair, idx)
                tokens.extend(part_tokens)
            
            # Insert special tokens in their original form
            if special_tokens:
                tokens.append(special_tok_map[special_tokens.pop(0)])
        
        return tokens
    

        