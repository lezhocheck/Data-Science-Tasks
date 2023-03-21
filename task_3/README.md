# Regression on the tabular data

## Task

    You have a dataset (internship_train.csv) that contains 53 anonymized features and a target column.  
    Your task is to build model that predicts a target based on the proposed features.  
    Please provide predictions for internship_hidden_test.csv file. Target metric is RMSE.

## Solution

    The solution contains:
        - data analysis jupyter notebook (regression.ipynb); 
        - code for modeling in Python3 (model.py);
        - predictions of the hidden test (predictions.csv).
        
    - An analysis of the data was carried out with feature extraction and it was concluded that the main influence on the target variable has feature #6.
    - In addition, data has many useless features, that need to be filtered and not included in the resulting model.
    - The resulting model is based on a desision tree regressor that can capture non-linear relationships and filter out useless features.

## Installation and usage

Make sure you have already installed **Python3** interpreter and **git**. You can then run the following commands to get the scripts on your computer:

```
git clone https://github.com/lezhocheck/quantum_ds_task.git
```

After cloning the repository, make sure you have installed all the dependencies for the conda environment (listed in *path_to_local_repository://task_3/requirements.txt* file).  

**Note: replace all *path_to_local_repository://* with the actual path to the local repository on your machine.**  

To install all these dependencies, just run:

```
pip install -r path_to_local_repository://task_3/requirements.txt
```

**After installing all the requirements, copy the *internship_hidden_test.csv* and *internship_train.csv* files to the *data/...* folder. Please, do not rename any of these files.**

### Model

To run model on the data, use:

```
python path_to_local_repository://task_3/model.py -data [path_to_file_with_data]
```

You will see the predictions in your console's standart output. 

**Note: please note that the *model.py* depends on *regression.save* and *scaler.save* files. So, if you change the location or delete these files, you should update the corresponding dependencies in the model.py file.**

To save the predictions to a .csv file, use:

```
python path_to_local_repository://task_3/model.py -data [path_to_file_with_data] --out [path_to_output_file]
```

### Jupyter notebook

To run **regression.ipynb**, start the jupyter kernel and re-run all cells.  
For more information see: https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html  

**Note: please note that the *regression.ipynb* depends on the contents of the */data/...* folder. So, if you change the location or rename it, you should update the corresponding dependencies in the *regression.ipynb.***
