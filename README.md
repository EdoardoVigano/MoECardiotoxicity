# README: INFERENCE MODELS
Official implementation of: Mixture of Experts for Multitask Learning in Cardiotoxicity Assessment [[1]](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-025-01072-7)

(required: anaconda)

# INSTALLATION

1. Download MoECardiotoxicity

2. create an environment 1: 
	env for models
	conda create --name alternative_multitask python=3.12
	conda activate alternative_multitask

	install torch version suitable for your pc: https://pytorch.org/get-started/locally/
	example: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

	pip install  scikit-learn==1.4.2 rdkit=='2024.09.4' pandas mordred

3. create an environment 2: 
	CDDD env: https://github.com/jrwnter/cddd
	(download cddd. the cddd folder should be in the MoECardiotoxicity)

4. how to use by terminal:
	* conda activate alternative_multitask 
	* enter in folder  

 	_batch_ 
	* [command] python inference.py --filename data_test_model.csv # file csv with named column as "smiles"

	_single molecule_
	* [command] python inference.py --smiles c1ccccc1O

5. results:
	prediction: csv file named prediction.csv in folder

# References
[1] Viganò, E.L., Iwan, M., Colombo, E. et al. Mixture of experts for multitask learning in cardiotoxicity assessment. J Cheminform 17, 135 (2025). https://doi.org/10.1186/s13321-025-01072-7


