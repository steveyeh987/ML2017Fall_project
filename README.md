# Machine Learning Final Project: Listen and Translate  
### Team: 溫妮孝周的老公  
  
# Scripts:  
- 2 of all:  
  - model.py: Dual LSTM based retrieval model implementation  
  - reproduce.py: Kaggle reproduce implementation  
  
# Requirements:  
- Python 3.5  
- gensim 3.1.0 
- keras 2.0.8 
- tensorflow 1.3.0 
- scikit-learn 0.19.1 
- pandas 0.20.3 
- numpy 1.13.3 
  
# Files:  
> All required files should be named as follows and be in the same folder as scripts:  
- train.data  
- train.caption  
- test.data  
- test.csv  
- model.h5  
  
# Descriptions of scripts:  
- model.py:  
  - Execution:  
  ```
    python3 model.py train.data test.data train.caption test.csv filename.csv  
  ```
  - Output:  
  ```
    Generate a keras model file (model.h5) and write the prediction to the file (filename.csv) at the end.  
  ```
	  
- reproduce.py:  
  - Execution:  
  ```
    python3 reproduce.py train.data test.data train.caption test.csv model.h5 filename.csv    
  ```
  - Output:  
  ```
    Reproduce the prediction to the file (filename.csv) at the end by loading the pretrained model.  
  ```
  
