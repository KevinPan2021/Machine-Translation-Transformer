application_name = 'Machine Translation'
# pyqt packages
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic

import sys
import torch

from transformer import Transformer
from main import compute_device
from tokenizer import Tokenizer
        

class QT_Action(QMainWindow):
    def __init__(self):
        # system variable
        super(QT_Action, self).__init__()
        uic.loadUi('qt_main.ui', self)
        
        self.setWindowTitle(application_name) # set the title
        self.mouse_pos = None
        
        # runtime variable
        self.model = None
        self.seq_length = 64
        
        # load tokenizer
        self.tokenizer = Tokenizer()
        self.tokenizer.load('vocab.pkl')
        
        # load the model
        self.load_model_action()
        

            
    # linking all button/textbox with actions    
    def link_commands(self,):
        self.comboBox_model.activated.connect(self.load_model_action)
        self.toolButton_process.clicked.connect(self.process_action)
        
     
            
    # choosing between models
    def load_model_action(self,):
        self.model_name = self.comboBox_model.currentText()
        
        # load the model
        if self.model_name == 'Transformer':
            pad_ind = self.tokenizer.get_special_token()['<|pad|>']
            start_ind = self.tokenizer.get_special_token()['<|startoftext|>']
            end_ind = self.tokenizer.get_special_token()['<|endoftext|>']
            
            seq_len = 64
            
            d_model = 640
            num_layers = 8
            num_heads = 10
            d_ff = 2048
            self.model = Transformer(
                self.tokenizer.vocab_size(), 
                pad_ind, start_ind, end_ind,
                seq_len, num_heads=num_heads, num_layers=num_layers,
                d_model=d_model, d_ff=d_ff
            )
            
        # loading the training model weights
        self.model.load_state_dict(torch.load(f'{self.model_name}.pth'))
            
        # move model to GPU
        self.model = self.model.to(compute_device())
        
        self.model.eval() # Set model to evaluation mode
    
        
    
        
    def process_action(self):
        # get the input sentence
        input_sentence = self.textEdit_input.toPlainText()
        
        # tokenize
        x = self.tokenizer.encode(input_sentence)
        special_tok = self.tokenizer.get_special_token()
        
        # add <|endoftext|> token to the end of x
        x = x[:self.seq_length-1] # final x shouldn't exceed seq_length
        x.append(special_tok['<|endoftext|>'])
        
        # Pad x and y to seq_length
        x = x + [special_tok['<|pad|>']] * (self.seq_length - len(x))
        
        # conver to tensor
        x = torch.tensor(x, dtype=torch.long).unsqueeze(0)
        
        # move to device
        x = x.to(compute_device())
        
        # model inference
        out_token = self.model.inference(x).to('cpu')
        
        # token decode
        out_sentence = self.tokenizer.decode(out_token.numpy(), omit_special_tok=True)
        
        self.textEdit_output.setPlainText(out_sentence)
        
        
def main():
    app = QApplication(sys.argv)
    action = QT_Action()
    action.link_commands()
    action.show()
    sys.exit(app.exec_())
    
    
if __name__ == '__main__':
    main()