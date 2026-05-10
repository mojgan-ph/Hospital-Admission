Analysis on UCI Diabetes 130-US Hospitals dataset (Strack et al. 2014):   
Link: archive.ics.uci.edu/dataset/296  
Citation: Strack et al. (2014), BioMed Research International, Article 781670.

**Claude code prompts:**

1- Complete the module split such that it splits the dataset into train, validation and test sets with ratios of 0.6, 0.2 and 0.2 respectively, and saves them separately in folder data/split.  

2- Change split.py such that it changes the target to have binary values before splitting, 1 if target=<30 and 0 otherwise.  

3- Change split.py to do a group-aware split, such that each patient_nbr is not in more than one of train, validation, and test set.   

4- Add a method to utils.py for loading the dataset, that takes 2 files names (features and target) and returns two dataframe, one for each. Before returning, do the following:
 - Find all categorical variables in file split/variables.csv 
 - Change all categorical features using X[feature].astype("category"). 

5- Update train.py to use load_dataset from utils module and train an xgboost classifier on the train set

6- Update train.py to use validation set as well, with early stopping enabled

7- Update split.py and utils.py to save the mapping of each categorical column to its full set of levels seen across the entire dataset (before splitting) and use it for data loading.

8- Write evaluate.py to evaluate the model using the test set and save the predictions and evaluation metrics in csv files

9- Write analyse.py to do the following: 
 - plot the feature importance of the model 
 - plot the confusion matrix

10- In evaluate.py, add code that creates ROC diagram

11- How do I know what was the probability threshold of considering a prediction positive on ROC for each point on the diagram? can you at least add this information for each 0.1 increase of y axis?

12- Update evaluate.py to generate metrics in form of a table, for all thresholds that are listed on the auc diagram, instead of current metrics that are saved in metrics.csv

13- Update analyse.py to create the confusion matrix for two probability thresholds in separate files: 0.1 and 0.2 

**To run:**
- (recommended) make a virtual environment and activate it
- run command: pip install -r .\requirements.txt
- run command: dvc repro