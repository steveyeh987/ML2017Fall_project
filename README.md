# Machine Learning Final Project: Listen and Translate  
### Team: 溫妮孝周的老公  
  
# Scripts:  
- 2 of all:  
  - model.py: Dual LSTM based retrieval model implementation  
  - reproduce.py: Kaggle reproduce implementation  
  
# Requirements:  
- Python 3  
- gensim  
- keras  
- tensorflow  
- scikit-learn  
- pandas  
- numpy  
  
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
    python3 model.py train.data test.data train.caption test.csv predict_path  
  ```
  - Output:  
  ```
    Generate a keras model file (model.h5) and write the prediction to the file (predict_path.csv) at the end.  
  ```
	  
- reproduce.py:  
  - Execution:  
  ```
    python3 reproduce.py model.h5 predict_path  
  ```
  - Output:  
  ```
    Write the prediction to the file (predict_path.csv) at the end by loading the pretrained model.  
  ```
  
