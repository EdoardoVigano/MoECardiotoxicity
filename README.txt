README: INFERENCE MODELS
official implementation of: Mixture of Experts for Multitask Learning in Cardiotoxicity Assessment

(required: anaconda)

INSTALLATION

1. Download MoECardiotoxicity

2. create an environment 1: 
	env for models
	conda create --name alternative_multitask python=3.12
	conda activate alternative_multitask

	# install torch version suitable for your pc: https://pytorch.org/get-started/locally/
	es my case: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

	pip install  scikit-learn==1.4.2 rdkit=='2024.09.4' pandas mordred

3. create an environment 2: 
	CDDD env: https://github.com/jrwnter/cddd
	(download cddd. the cddd folder should be in the MoECardiotoxicity)

3. how to use by terminal:
	* conda activate alternative_multitask 
	* enter in folder  

	# batch 
	* [command] python inference.py --filename data_test_model.csv # file csv with named column as "smiles"

	# single molecule
	* [command] python inference.py --smiles c1ccccc1O

4. results:
	prediction: csv file named prediction.csv in folder

