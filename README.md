#Multi-Instance-Experiments

By Gitte Vanwinckelen (gitte.vanwinckelen at cs.kuleuven.be)

##Overview

This repository contains the Python implementation for the experiments in:
>''Instance-level accuracy versus bag-level accuracy in multi-instance learning''. Gitte Vanwinckelen, Vinicius Tragante Do O, Daan Fierens and Hendrik Blockeel. Data Mining and Knowledge Discovery (accepted 2014)

##Dependencies

The experiments were tested on Ubuntu 12.04 with the following software installed:
* [Python 2.7] (https://www.python.org/download/releases/2.7/)
* [Numpy] (http://www.numpy.org/) 1.8.1
* [Scipy] (www.scipy.org) 0.13.3
* [Matplotlib] (matplotlib.org) 1.3.1
* [pyPdf] (http://pybrary.net/pyPdf/) 1.13

##Datasets

The sources of the datasets described in Section 4.1.1:
* SIVAL: [AMIL: Active Multiple-Instance Library](http://pages.cs.wisc.edu/~bsettles/amil/)
* Newsgroup: [Text Data for Multi-Instance Learning by LAMBDA, Nanjing University](http://lamda.nju.edu.cn/data_MItext.ashx)
* The [UCI repository] (http://archive.ics.uci.edu/ml/)

The SIVAL datasets can be used with the Python scripts as downloaded from the 'AMIL: Active Multiple-Instance Library' webpage above. A preprocessed version of the Newsgroup and UCI multi-instance datasets can be found in the 'Data' folder of this repository.

##Overview

The experimental procedure is as follows:
 1. Each original dataset is split into ten partitions that are balanced on the bag-level. The datasets are saved in ARFF format.
 2. Each learner is evaluated using 10-fold cross-validation. For each of the 10 evaluations, 9 partitions are merged into a single dataset and the leftover fold is used as a test fold. For each of the 10 runs, the individual prediction probabilities together with the corresponding correct labels are saved in a separate file. Two types of experiments are run:
  1. Bag-level and instance-level performance of the multi-instance learners.
  2. Performance of single-instance learners on one-sided noisy data.
 3. The 10-fold cross-validation results are used to compute performance metrics: Accuracy, AUC, and weighted accuracy (0.5*(TPR+TNR)).
 4. The plots and tables in the paper are generated:
  1. Section 4.2.1: A visualization of the rank correlation between bag-level and instance-level performance of the multi-instance algorithms.
  2. Section 4.2.2: Latex tables for the probabilistic analysis.
  3. Section 4.3.2: Scatter plots comparing bag-level and instance-level performance of the multi-instance algorithms for each learner over all datasets.
  4. Section 5: Scatter plots comparing instance-level performance of the multi-instance algorithms with the performance of the single-instance algorithms for each learner combination (Table 9) over all datasets.

##Running

### Preparing the SIVAL data for 10-fold cross-validation

 1. Run the script 'toArffSIVAL.py' to convert the SIVAL data to ARFF format. In the script, use the run function as described by the docstring. The input are the original SIVAL datasets as downloaded from the 'AMIL: Active Multiple-Instance Library'
 2. Run 'createDatasetsSIVAL.py'. This script can be run from the terminal. For further instructions see the help and usage message of the script. 

Four types of datasets are created, each in their own subdirectory:
* **BagData_OnePerBag** contains the ARFF multi-instance datasets with a single instance per bag.
* **BagData_Regular** contains the ARFF multi-instance datasets with multiple instances per bag.
* **SingleInstance_CorrectLabels** contains single-instance ARFF datasets, where the instances are correctly labeled. 
* **SingleInstance_Noisy** contains single-instance ARFF datasets where the instances are labeled with the label from the bag from which they originate. These are the 'one-sided' noisy datasets from Section 5.
 
### Preparing the Newsgroup and UCI datasets for 10-fold cross-validation

Run the script 'createDatasets_UCI_Text.py' from the terminal. To create the UCI datasets set 'datasetType' to 'uci'. To create the Newsgroup datasets set it to 'text'. Similarly as for the SIVAL datasets, four different types of datasets are created, each in their own subdirectory.

    python ./createDatasets_UCI_Text.py --inDir  '/home/me/OriginalDataFolder' --outDir '/home/me/outputFolder' --datasetType 'uci'   

For further instructions see the help and usage message of the script.

### Evaluation of the algorithms

Run the script 'evaluation.py' from the terminal. Two examples are given below. To compare bag-level and instance-level performance, the multi-instance algorithms are in both cases trained on the same dataset, and evaluated on the same test set that is presented in two different forms: one with regular bags, and one with a single instance per bag. This is accomplished by specifying two data paths to the script: 'dataPath' contains the datasets that are used for training and testing. 'dataPath2' contains the data in the second test format. A similar procedure is followed for the experiments comparing single-instance with multi-instance learners. Two usage examples are given below.

    python ./evaluation.py --labels 'talk_politics_guns' --datatype 'BagLevel' --outPath '/home/me/experimentResults' --dataPath '/home/me/data_text/BagData_Regular' --dataPath2 '/home/me/data_text/BagData_OnePerBag' --classifiers 'MILR' 'MIRI' --folds 1 2 3 4 5

    python ./evaluation.py --datatype 'Noisy' --outPath '/home/me/experimentResults' --dataPath '/home/me/data_text/SingleInstance_Noisy' --dataPath2 '/home/me/data_text/SingleInstance_CorrectLabels'

For further instructions see the help and usage message of the script.

### Postprocessing

Computing performance metrics (Accuracy, AUC and weighted accuracy) is accomplished with 'processExperimentResults.py'. Example:

    python ./processExperimentResults.py --inDir '/home/me//Results/raw_uci_baglevel/Path1'  --outDir  '/home/me/Results/processed_uci/BagData_Regular' --datasetType 'uci' --algorithmType 'BagLevel'

### Creating plots and tables

 1. Section 4.2.1: Visualization of rank correlation between bag-level and instance-level performance: **linkedColumns.py**.
 
 2. Section 4.2.2: Latex tables for the probabilistic analysis: **probabilisticAnalysis.py**
 
 3. Section 4.3.2: Scatter plots comparing bag-level and instance-level performance: **scatter.py**
 
 4. Section 5: Scatter plots comparing performance of multi-instance and single-instance algorithms: **scatter.py**

See the docstrings for the usage instructions of these scripts.
   
