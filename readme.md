Analysis on UCI Diabetes 130-US Hospitals dataset (Strack et al. 2014):   archive.ics.uci.edu/dataset/296
Citation: Strack et al. (2014), BioMed Research International, Article 781670.

Claude code prompts:
1- Complete the module split such that it splits the dataset into train, validation and test sets with ratios of 0.6, 0.2 and 0.2 respectively, and saves them separately in folder data/split.
2- Change split.py such that it changes the target to have binary values before splitting, 1 if target=<30 and 0 otherwise.
3- Change split.py to do a group-aware split, such that each patient_nbr is not in more than one of train, validation, and test set. 
4- Add a method to utils.py for loading the dataset, that takes 2 files names (features and target) and returns two dataframe, one for each. Before returning, do the following:
 - Find all categorical variables in file split/variables.csv 
 - Change all categorical features using X[feature].astype("category").
5- Update train.py to use load_dataset from utils module and train an xgboost classifier on the train set
6- Update train.py to use validation set as well, with early stopping enabled
7- Update split.py and utils.py to save the mapping of each categorical column to its full set of levels seen across the entire dataset (before splitting) and use it for data loading. 