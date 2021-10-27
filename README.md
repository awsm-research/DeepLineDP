
# Supplementary Materials for "DeepLineDP: Towards a Deep Learning Approach for Line-Level Defect Prediction"
  

## Datasets

The datasets are obtained from Wattanakriengkrai et. al. The datasets contain 32 software releases across 9 software projects. The datasets that we used in our experiment can be found in this [github](https://github.com/awsm-research/line-level-defect-prediction).

The file-level datasets (in the File-level directory) contain the following columns

 - `File`: A file name of source code
 - `Bug`: A label indicating whether source code is clean or defective
 - `SRC`: A content in source code file

The line-level datasets (in the Line-level directory) contain the following columns
 - `File`: A file name of source code
 - `Commit`: A commit id of bug-fixing commit of the file
 - `Line_number`: A line number where source code is modified
 - `SRC`: An actual source code that is modified

For each software project, we use the oldest release to train DeepLineDP models. The subsequent release is used as validation sets. The other releases are used as test sets.

For example, there are 5 releases in ActiveMQ (e.g., R1, R2, R3, R4, R5), R1 is used as training set, R2 is used as validation set, and R3 - R5 are used as test sets.
  
## Repository Structure

Our repository contains the following directory

 - `output`: This directory contains the following sub-directories:
	 - `loss`: This directory stores  training and validation loss
	 - `model`: This directory stores trained models
	 - `prediction`: This directory stores prediction (in CSV files) obtained from the trained models
	 - `Word2Vec_model`: This directory stores word2vec models of each software project
 - `script`: This directory contains the following directories and files:
	 - `preprocess_data.py`: The source code used to preprocess datasets for file-level model training and evaluation
	 - `export_data_for_line_level_baseline.py`: The source code used to prepare data for line-level baseline
	 - `my_util.py`: The source code used to store utility functions
	 - `train_word2vec.py`: The source code used to train word2vec models
	 - `DeepLineDP_model.py`: The source code that stores DeepLineDP architecture
	 - `train_model.py`: The source code used to train DeepLineDP models
	 - `generate_prediction.py`: The source code used to generate prediction (for RQ1-RQ3)
	 - `generate_prediction_cross_projects.py`: The source code used to generate prediction (for RQ4) 
	 - `get_evaluation_result.R`: The source code used to generate figures for RQ1-RQ3, and show RQ4 result
	 - `file-level-baseline`: The directory that stores implementation of the file-level baselines, and `baseline_util.py` that stores utility function of the baselines
	 - `line-level-baseline`: The directory that stores implementation of the line-level baselines

## Environment Setup


### Python Environment Setup
1. clone the github repository by using the following command:

		git clone https://github.com/awsm-research/DeepLineDP.git

2. download the dataset from this [github](https://github.com/awsm-research/line-level-defect-prediction) and keep it in `./datasets/original/`

3. use the following command to install required libraries in conda environment

		conda env create -f requirements.yml
		conda activate DeepLineDP_env

4. install PyTorch library by following the instruction from this [link](https://pytorch.org/) (the installation instruction may vary based on OS and CUDA version)


### R Environment Setup

Download the following package: `tidyverse`, `gridExtra`, `ModelMetrics`, `caret`, `reshape2`, `pROC`, `effsize`, `ScottKnottESD`


## Experiment

### Experimental Setup

We use the following hyper-parameters to train our DeepLineDP model

- `batch_size` = 32
- `num_epochs` = 10
- `embed_dim (word embedding size)` = 50
- `word_gru_hidden_dim` = 64
- `sent_gru_hidden_dim` = 64
- `word_gru_num_layers` = 1
- `sent_gru_num_layers` = 1
- `dropout` = 0.2
- `lr (learning rate)` = 0.001

### Data Preprocessing

1. run the command to prepare data for file-level model training. The output will be stored in `./datasets/preprocessed_data`

		python preprocess_data.py

2. run the command to prepare data for line-level baseline. The output will be stored in `./datasets/ErrorProne_data/` (for ErrorProne), and `./datasets/n_gram_data/` (for n-gram)

		python export_data_for_line_level_baseline.py

### Word2Vec Model Training

To train Word2Vec models, run the following command:

	python train_word2vec.py <DATASET_NAME>
  
Where \<DATASET_NAME\> is one of the following: `activemq`, `camel`, `derby`, `groovy`, `hbase`, `hive`, `jruby`, `lucene`, `wicket`

### DeepLineDP Model Training and Prediction Generation

To train DeepLineDP models, run the following command:

	python train_model.py -dataset <DATASET_NAME>
  

The trained models will be saved in `./output/model/DeepLineDP/<DATASET_NAME>/`, and the loss will be saved in `../output/loss/DeepLineDP/<DATASET_NAME>-loss_record.csv`

To make a prediction of each software release, run the following command:

	python generate_prediction.py -dataset <DATASET_NAME>

The generated output is a csv file which contains the following information:

 - `project`: A software project, as specified by \<DATASET_NAME\>
 - `train`: A software release that is used to train DeepLineDP models
 - `test`: A software release that is used to make a prediction
 - `filename`: A file name of source code
 - `file-level-ground-truth`: A label indicating whether source code is clean or defective
 - `prediction-prob`: A probability of being a defective file
 - `prediction-label`: A prediction indicating whether source code is clean or defective
 - `line-number`:  A line number of a source code file
 - `line-level-ground-truth`: A label indicating whether the line is modified 
 - `is-comment-line`: A flag indicating whether the line is comment
 - `token`: A token in a code line  
 - `token-attention-score`: An attention score of a token

The generated output is stored in `./output/prediction/DeepLineDP/within-release/`


To make a prediction across software project, run the following command:
	
	python generate_prediction_cross_projects.py -dataset <DATASET_NAME>
	
The generated output is a csv file which has the same information as above, and is stored in `./output/prediction/DeepLineDP/cross-project/`

### File-level Baseline Implementation

There are 4 baselines in the experiment (i.e., `Bi-LSTM`, `CNN`, `DBN` and `BoW`). To train the file-level baselines, go to `./script/file-level-baseline/` then run the following commands

 - `python Bi-LSTM-baseline.py -data <DATASET_NAME> -train`
 - `python CNN-baseline.py -data <DATASET_NAME> -train`
 - `python DBN-baseline.py -data <DATASET_NAME> -train`
 - `python BoW-baseline.py -data <DATASET_NAME> -train`

The trained models will be saved in `./output/model/<BASELINE>/<DATASET_NAME>/`, and the loss will be saved in `../output/loss/<BASELINE>/<DATASET_NAME>-loss_record.csv`

where \<BASELINE\> is one of the following: `Bi-LSTM`, `CNN`, `DBN` or `BoW`.

To make a prediction, run the following command:
 - `python Bi-LSTM-baseline.py -data <DATASET_NAME> -predict -target_epochs 6`
 - `python CNN-baseline.py -data <DATASET_NAME> -predict -target_epochs 6`
 - `python DBN-baseline.py -data <DATASET_NAME> -predict`
 - `python BoW-baseline.py -data <DATASET_NAME> -predict`

The generated output is a csv file which contains the following information:

 - `project`: A software project, as specified by \<DATASET_NAME\>
 - `train`: A software release that is used to train DeepLineDP models
 - `test`: A software release that is used to make a prediction
 - `filename`: A file name of source code
 - `file-level-ground-truth`: A label indicating whether source code is clean or defective
 - `prediction-prob`: A probability of being a defective file
 - `prediction-label`: A prediction indicating whether source code is clean or defective

The generated output is stored in `./output/prediction/<BASELINE>/`

### Line-level Baseline Implementation

There are 2 baselines in this experiment (i.e., `N-gram` and `ErrorProne`). 

To obtain the result from `N-gram`, go to `/script/line-level-baseline/ngram/` and run code in `n_gram.java`. The result will be stored in `/n_gram_result/` directory. After all results are obtained, copy the `/n_gram_result/` directory to the `/output/` directory.

To obtain the result from `ErrorProne`, go to `/script/line-level-baseline/ErrorProne/` and run code in `run_ErrorProne.ipynb`. The result will be stored in `/ErrorProne_result/` directory. After all results are obtained, copy the `/ErrorProne_result/` directory to the `/output/` directory.

### Obtaining the Evaluation Result 

Run `get_evaluation_result.R` to get the result of RQ1-RQ4 (may run in IDE or by the following command)

	Rscript  get_evaluation_result.R

The results are figures that are stored in `./output/figures/`
