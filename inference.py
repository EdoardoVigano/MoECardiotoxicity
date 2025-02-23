## inference

import pandas as pd
import numpy as np
import pickle, os, argparse, subprocess, sys
import torch
# from model import multigate_multimodal
from model import multigate_multimodal
from datetime import datetime
# rdkit
from rdkit import Chem
from mordred import Calculator, descriptors, error

def check_smiles(smiles):
    try: Chem.MolFromSmiles(smiles)
    except: 
        print(f"invalid smiles {smiles}.")
        sys.exit(1)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='cardiotoxicity assessment')

    # Add an argument for input
    parser.add_argument('--smiles', required=False, help='Specify target smiles')
    parser.add_argument('--filename', required=False, help='Specify csv files with "smiles" as columns header')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the input value
    input_value = args.smiles
    if input_value is None:
        input_value = args.filename

    # Your script logic here
    print(f"Input value: {input_value}")
    return input_value

def run_script_in_conda_env(smiles:str, singlesmiles=True):
    # Combine activation, script execution, and deactivation in a single command
    # Specify the path to your Python script in the other environment
    """
    You must env the conda env defined by https://github.com/jrwnter/cddd
    """
    script_path = os.path.join(os.getcwd(), "cddd", "cddd_calculation.py")
    conda_env = 'cddd'
    if singlesmiles: 
        print("SINGLE MOLECULE")
        command = f"conda activate {conda_env} && python {script_path} --smiles {smiles} && conda deactivate"
    else:  
        print("DATASET OF MOLECULEs")
        command = f"conda activate {conda_env} && python {script_path} --file {smiles} && conda deactivate"
    # cddd --input smiles.smi --output descriptors.csv  --smiles_header smiles
    # Run the entire command
    subprocess.run(command, shell=True, check=True)

def cddd_calculation(smiles, singlesmiles=True):
    """
    You have to use subprocess to calcualte cddd descriptors
    in other environment.
    """
    # Run the script in the other virtual environment with the input argument
    try: 
        run_script_in_conda_env(smiles=smiles, singlesmiles=singlesmiles)
        path = os.path.join(os.getcwd(), "smiles_CDDD.csv")
        data = pd.read_csv(path)
        os.remove(path)
        return data
    except: 
        print("error in subprocess")

def mordred_calculator(dataset:pd.DataFrame):
    # file = os.path.join(os.getcwd(), 'model', 'MDs_name.pickle')
    # file2 = os.path.join(os.getcwd(), 'model', 'MD_iqr_scaler_all_data.pickle')
    file2 = os.path.join(os.getcwd(), 'model', 'my_pipeline.pkl')
    
    # with open(file, "rb") as f:
        # cols_ = pickle.load(f) # cols from mordred
        # cols_homade = [f"MDs_{i}" for i in cols_]

    with open(file2, "rb") as f:
        scaler = pickle.load(f)

    mols = [Chem.MolFromSmiles(smi) for smi in dataset['smiles']]
    calc = Calculator(descriptors, ignore_3D=True)
    # as pandas
    df = calc.pandas(mols)
    df = pd.concat([dataset, df], axis=1)

    df = df.applymap(lambda x: np.nan if isinstance(x, error.Error) or isinstance(x, error.Missing) else x) # remove errors using nan
    df_selected = df.loc[:, scaler.feature_names_in_]
    # df_selected.columns = cols_homade
    return scaler.transform(df_selected)# pd.DataFrame(scaler.transform(df.loc[:, cols_homade]), columns = scaler.get_feature_names_out()).to_numpy()

def inference(models:dict, smiles, target_dict:dict, singlemol=True):

    # Prediction
    results_final = {}
    for key, model in models.items():
        results_final[key] = model.predict(target_dict[key])

    if singlemol: return pd.DataFrame(pd.DataFrame(results_final, index=[smiles]))
    else: return pd.DataFrame(pd.DataFrame([results_final])) #, index=smiles)

def import_models():

    "Import models find in folder"

    path = os.path.join(os.getcwd(), "model", "multitask_cardiotox_model.pth")
    state_dict = torch.load(path)

    # Set the model to evaluation mode (if using for inference)
    # loaded_model.eval()
    input_parameters = {'branch': {'NLPcustom': 0, 'CDDD': 1, 'morgan': 0, 'MDs': 1, 'ChemBERTa': 0},
            'gate_dimension': 64,
            'input_dim_NN': {'MDs': 494, 'CDDD': 512, 'ChemBERTa': 768, 'morgan': 1024},
            'task_name': ['labels_Apical cardiotoxicity',
            'labels_Aryl hydrocarbon receptor',
            'labels_Cardiomyocyte Myocardial Injury',
            'labels_Change Action Potential',
            'labels_Change in Inotropy',
            'labels_Change In Vasoactivity',
            'labels_Endothelial injury coagulation',
            'labels_hERG channels inhibitors',
            'labels_Increase mitochondrial dysfunction',
            'labels_Inhibition mitochondrial complexes',
            'labels_OxidativeStress',
            'labels_Valvular Injury Proliferation'],
            'nlp': {'emb_dim': 512,
            'vocab_dim': 45,
            'hidden_size_convs': [256, 128],
            'hidden_size_lstm': {'hidden_size': 64, 'num_layers': 3},
            'dropoutprob': 0.3},
            'NN': {'hidden_size': [256, 128], 'dropoutprob': 0.3}}
    
    loaded_model = multigate_multimodal(input_parameters=input_parameters)

    loaded_model.load_state_dict(state_dict) 

    thr_path = os.path.join(os.getcwd(), "model", "thr.pkl")

    with open(thr_path, 'rb') as file:
         thr = pickle.load(file)

    return loaded_model.eval(), thr

def localOutlierFactor_applicability_domain(test):
        
        """
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor
        
        The anomaly score of each sample is called the Local Outlier Factor. 
        It measures the local deviation of the density of a given sample with respect to its neighbors. 
        It is local in that the anomaly score depends on how isolated the object is with respect to the surrounding neighborhood. 
        More precisely, locality is given by k-nearest neighbors, whose distance is used to estimate the local density. 
        By comparing the local density of a sample to the local densities of its neighbors, one can identify 
        samples that have a substantially lower density than their neighbors. These are considered outliers.

        pfloat, default=2
        Parameter for the Minkowski metric from sklearn.metrics.pairwise_distances. 
        When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. 
        For arbitrary p, minkowski_distance (l_p) is used.

        NOTE:
        noveltybool, default=False
        By default, LocalOutlierFactor is only meant to be used for outlier detection (novelty=False). 
        Set novelty to True if you want to use LocalOutlierFactor for novelty detection. 
        In this case be aware that you should only use predict, decision_function and score_samples on new unseen data and not on the training set; 
        and note that the results obtained this way may differ from the standard LOF results."""
        
        clf_path = os.path.join(os.getcwd(), "model", "clf.pkl")

        with open(clf_path, "rb") as f:
            clf = pickle.load(f)

        return ['in' if i==1 else 'out' for i in clf.predict(test).tolist()]

if __name__ == "__main__":

    tasks_names = ['Apical cardiotoxicity',
                    'Aryl hydrocarbon receptor',
                    'Cardiomyocyte Myocardial Injury',
                    'Change Action Potential',
                    'Change in Inotropy',
                    'Change In Vasoactivity',
                    'Endothelial injury coagulation',
                    'hERG channels inhibitors',
                    'Increase mitochondrial dysfunction',
                    'Inhibition mitochondrial complexes',
                    'OxidativeStress',
                    'Valvular Injury Proliferation']
    
    # ask smiles of chemicals to evaluate
    smiles = main()
    if ".csv" in smiles:
        print("file to manage:")
        data = pd.read_csv(smiles)
        for smi in data['smiles']:
            check_smiles(smi)
    else:
        print(f"SMILES to manage: {smiles}")
        check_smiles(smiles)
        data = pd.DataFrame([smiles], columns=['smiles'])
        

    # import models pipeline and descriptors
    print("[INFO]: Models import...")
    model, thr = import_models()
    print("done")
    
    print("[INFO]: calculate CDDD descriptors for target...")
    target_CDDD = cddd_calculation(smiles, singlesmiles=False)
    print(f"{target_CDDD}done")

    data_test_cddd = torch.tensor(target_CDDD.drop(['original_smiles', 'SMILES'], axis=1).values, dtype=torch.float32)
    
    print("[INFO]: calculate MDs descriptors for target...")
    data_test_MSs = torch.tensor(mordred_calculator(data), dtype=torch.float32)

    x = {'CDDD': data_test_cddd, "MDs": data_test_MSs}

    prediction = model(x)
    if prediction.shape[0] == 1:
        results = pd.DataFrame((torch.sigmoid(prediction) > thr).numpy().squeeze().astype(int)).T
        # results = pd.DataFrame((torch.sigmoid(prediction) > thr).numpy().squeeze().astype(int), columns=tasks_names)
        results.columns = tasks_names
    else:
        results = pd.DataFrame((torch.sigmoid(prediction) > thr).numpy().squeeze().astype(int), columns=tasks_names)

    results_ = pd.concat([target_CDDD.loc[:, ['original_smiles', 'SMILES']], results], axis=1)
    results_['ApplicabilityDomain'] = localOutlierFactor_applicability_domain(target_CDDD.drop(['original_smiles', 'SMILES'], axis=1).values)
    
    # remove prediction for the task that don't reach satisfactory results
    # results_.drop(['Inhibition mitochondrial complexes'], axis=1, inplace=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(os.getcwd(), 'results', f"results_{timestamp}.csv")
    results_.to_csv(path)

    print("done.. you can find the results in results.csv file")