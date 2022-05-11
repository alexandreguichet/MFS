# MFS
Modular Feature Selection (Mutual-Information-based Feature Selection)

This toolbox proposes mutual-information-based feature-selection modules to use in any machine-learning/deep-learning applications. 

The mutual information is computed using Kraskov/Ross/Gao's k-neighbours approach. 

This is a simple function that take N-by-Mx features with N-by-My labels and returns a Mx-by-My pd.DataFrame containing all features/labels mi. 

The next focus would be: 
- Perform feature-selection by minimising feature-redundancies and maximizing mutual-information.
- Implement partial-mutual-information / conditional-mutual-information

Run script: 
- check mifs_example.py for examples on hwo to use the framework for different tasks
- check mifs.py and mutual_information.py's documentation on all parameter uses and possibilities

Usage Example: 
```python
mifs = MIFS()
mifs.load_file("datasets\\IPODataFull.csv")

# extract features/labels
features = mifs.df.drop(columns = ["Survived"])
labels = mifs.df["Survived"].to_frame()

#Convert categorical
cat_columns = features.select_dtypes(['category', 'object']).columns
features[cat_columns] = features[cat_columns].astype('category').apply(lambda x: x.cat.codes)

#Calculate mutual information
#results is a dictionary!
# check results['selected'] for the final answer: unique features with most redundancies removed (final value is normalized)
# check results['threshold'] for all features above a threshold (n = 50 here), redundancies are still present
# check results['all'] for the mutual information matrix of the 50 selected features (that are above a threshold)
# check results['labels'] for the mutual information value of all features
results = mifs.select_n_features(n = 50, downsample = True) 
```

References: 

- B.C. Ross, "Mutual Information between Discrete and Continuous Data Sets". PloS ONE 9(2), 2014. 
	https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357
	
- A. Kraskov, H. Stogbauer, and P. Grassberger, "Estimating Mutual Information". Phys. Rev. E 69, 2003. 
	https://arxiv.org/abs/cond-mat/0305641
	
- W. Gao, S. Kannan, S. Oh, P. Viswanath, "Estimating Mutual Information for discrete-continuous mixtures". arXiv preprint, arXiv:1709.06212, 2017.
	https://arxiv.org/pdf/1709.06212.pdf
