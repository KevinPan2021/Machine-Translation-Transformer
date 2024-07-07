Introduction:
	This project aims to preform Machine Translation (English to Chinese) using Transformer Trained from Scratch.



Dataset: 
	https://www.kaggle.com/code/concyclics/machine-translation-between-chinese-and-english/



Build: 
	System:
		CPU: Intel i9-13900H (14 cores)
		GPU: NIVIDIA RTX 4060 (VRAM 8 GB)
		RAM: 32 GB

	Configuration:
		CUDA 12.1
		Anaconda 3
		Python = 3.11.7
		Spyder = 5.4.3
		
	Core Python Package:
		pytorch = 2.1.2
		numpy = 1.26.4
		matplotlib = 3.8.0
		pandas = 1.5.3
		tqdm = 4.64.1
		jieba = 0.42.1
		nltk = 3.8.1



Generate ".py" file from ".ui" file:
	1) open Terminal. Navigate to directory
	2) Type "pyuic5 -x qt_main.ui -o qt_main.py"



Core Project Structure:
	GUI.py (Run to generate a GUI)
	main.py (Run to train model)
	transformer.py
	qt_main.py
	training.py
	visualization.py


Credits:
	https://github.com/openai/gpt-2/blob/master/src/encoder.py
	