Introduction:
	This project aims to preform Machine Translation using Seq2Seq Attention and Seq2Seq Transformer from Scratch.



Dataset: 
	https://www.kaggle.com/code/concyclics/machine-translation-between-chinese-and-english/



Build: 
	M1 Macbook Pro
	Miniforge 3 (Python 3.9)
	PyTorch version: 2.2.1

* Alternative Build:
	Windows (NIVIDA GPU)
	Anaconda 3
	PyTorch



Generate ".py" file from ".ui" file:
	1) open Terminal. Navigate to directory
	2) Type "pyuic5 -x qt_main.ui -o qt_main.py"



Core Project Structure:
	GUI.py (Run to generate a GUI)
	main.py (Run to train model)
	model_transformer.py
	model_attention.py
	qt_main.py
	training.py
	visualization.py


Credits:
	Seq2Seq model is referenced from "https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html"
	Transformer model is referenced from "https://github.com/devjwsong/transformer-translator-pytorch/"
	