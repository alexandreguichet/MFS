# MFS
Modular Feature Selection (Mutual-Information-based Feature Selection)

This toolbox proposes mutual-information-based feature-selection modules to use in any machine-learning/deep-learning applications. 

The mutual information is computed using Gao's neighbours approach. 

This is a simple function that take N-by-Mx features with N-by-My labels and returns a Mx-by-My pd.DataFrame containing all features/labels mi. 

The next focus would be: 
- Perform feature-selection by minimising feature-redundancies and maximizing mutual-information.
- Implement partial-mutual-information / conditional-mutual-information


References: 

- B.C. Ross, "Mutual Information between Discrete and Continuous Data Sets". PloS ONE 9(2), 2014. 
	https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357
	
- A. Kraskov, H. Stogbauer, and P. Grassberger, "Estimating Mutual Information". Phys. Rev. E 69, 2003. 
	https://arxiv.org/abs/cond-mat/0305641
	
- W. Gao, S. Kannan, S. Oh, P. Viswanath, "Estimating Mutual Information for discrete-continuous mixtures". arXiv preprint, arXiv:1709.06212, 2017.
	https://arxiv.org/pdf/1709.06212.pdf
