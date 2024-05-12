application_name = 'Machine Translation'
# pyqt packages
from PyQt5.QtWidgets import QMainWindow, QApplication

import sys
import pickle
import torch
import jieba

from transformer import Transformer
from qt_main import Ui_Application
from main import BidirectionalMap, compute_device, inference, en_tokenizer, tokensToTensor


# Redirect output to os.devnull
jieba.default_logger.setLevel(20)

        

class QT_Action(Ui_Application, QMainWindow):
    def __init__(self):
        # system variable
        super(QT_Action, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.setWindowTitle(application_name) # set the title
        self.mouse_pos = None
        
        # runtime variable
        self.predicted = None
        self.image = None
        self.model = None
        self.input_tokenizer = None
        self.output_tokenizer = None
        self.input_vocab = None
        self.output_vocab = None
        self.max_len = 64
        
        # load language vocab
        self.load_language_action()
        
        # load the model
        self.load_model_action()
        

            
    # linking all button/textbox with actions    
    def link_commands(self,):
        self.comboBox_model.activated.connect(self.load_model_action)
        self.comboBox_input.activated.connect(self.load_language_action)
        self.comboBox_output.activated.connect(self.load_language_action)
        self.toolButton_process.clicked.connect(self.process_action)
        
        
    def load_language_action(self,):
        # get the input and output languages
        input_language = self.comboBox_input.currentText()
        output_language = self.comboBox_output.currentText()
        
        # load the vocabulary
        with open(f'{input_language}.pkl', 'rb') as f:    
            self.input_vocab = pickle.load(f)

        with open(f'{output_language}.pkl', 'rb') as f:    
            self.output_vocab = pickle.load(f)
        
        # load the tokenizer
        if input_language == 'English':
            self.input_tokenizer = en_tokenizer
        elif input_language == 'Chinese':
            self.input_tokenizer = jieba
        
        if output_language == 'English':
            self.output_tokenizer = en_tokenizer
        elif output_language == 'Chinese':
            self.output_tokenizer = jieba
                
            
    # choosing between models
    def load_model_action(self,):
        self.model_name = self.comboBox_model.currentText()
        src_pad_ind = self.input_vocab.get_value('<pad>')
        trg_pad_ind = self.output_vocab.get_value('<pad>')
        trg_sos_ind = self.output_vocab.get_value('<sos>')
        trg_eos_ind = self.output_vocab.get_value('<eos>')
        
        # load the model
        if self.model_name == 'Transformer':
            self.model = Transformer(len(self.input_vocab), len(self.output_vocab), src_pad_ind, trg_pad_ind, \
                                       trg_sos_ind, trg_eos_ind, self.max_len, num_layers=6)
         
        # loading the training model weights
        self.model.load_state_dict(torch.load(f'{self.model_name}.pth'))
            
        # move model to GPU
        self.model = self.model.to(compute_device())
        
        self.model.eval() # Set model to evaluation mode
    
        
    
        
    def process_action(self):
        # get the input sentence
        input_sentence = self.textEdit_input.toPlainText()
        
        # convert sentence to token
        X = self.input_tokenizer(input_sentence)
        
        # convert to tensor
        X = tokensToTensor(self.input_vocab, X, self.max_len)[1]
        
        # model inference
        out_sentence = inference(self.model, X.unsqueeze(0), self.output_vocab, compute_device(), self.max_len)
        
        # print out the output sentence
        if self.comboBox_output.currentText() == 'Chinese':
            out_sentence = ''.join(out_sentence)
        elif self.comboBox_output.currentText() == 'English':
            out_sentence = ' '.join(out_sentence)
        self.textEdit_output.setPlainText(out_sentence)
        
        
def main():
    app = QApplication(sys.argv)
    action = QT_Action()
    action.link_commands()
    action.show()
    sys.exit(app.exec_())
    
    
if __name__ == '__main__':
    main()