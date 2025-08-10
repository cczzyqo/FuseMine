import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
from transformers import AutoTokenizer, RobertaModel, AutoModelForMaskedLM
import os

def preparedata(csv_path, smiles_column):

    f = pd.read_csv(csv_path)
    N = f[smiles_column].values.shape[0]
    compounds = []
    
    print("Sanitizing SMILES...")
    for i in tqdm(range(N), desc="Processing SMILES"):
        mol = Chem.MolFromSmiles(f[smiles_column].values[i])
        if mol is not None:  
            smiles = Chem.MolToSmiles(mol)
            compounds.append(smiles)
        else:
            print(f"Warning: Invalid SMILES at index {i}")
    
    return compounds, len(compounds)

def get_smiles_embeddings(csv_path, smiles_column, save_path, device):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    

    device = torch.device(device)
    print(f"Using device: {device}")
    

    model_name = "./ChemBERTa-77M-MTR"
    # TODO  add_pooling_layer=False？ num_labels=2 ？
    model = RobertaModel.from_pretrained(model_name, num_labels=2, add_pooling_layer=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = model.eval().to(device)

    compounds, N = preparedata(csv_path, smiles_column)
    print(f"Total valid SMILES: {N}")
    

    all_embeddings = []
    

    for i in tqdm(range(0, len(compounds)), desc="Generating embeddings"):

        smiles = compounds[i]

        encodings = tokenizer(smiles, return_tensors='pt', 
                            padding="max_length", 
                            max_length=290, 
                            truncation=True)
        encodings = encodings.to(device)
        

        with torch.no_grad():
            output = model(**encodings)
            smiles_embedding = output.last_hidden_state[0, 0, :]
            smiles_embedding = smiles_embedding.cpu().numpy()
            all_embeddings.append(smiles_embedding)
    

    embeddings_array = np.array(all_embeddings, dtype = np.float64)
    

    np.save(save_path, embeddings_array)

    
    return embeddings_array

if __name__ == "__main__":

    csv_path = "smiles.csv"  
    smiles_column = "smiles"  
    
    # _, n = preparedata(csv_path=csv_path)
    # print(n)
    # assert False

    # embeddings = get_smiles_embeddings(
    #     csv_path=csv_path,
    #     smiles_column=smiles_column,
    #     save_path='./data/smiles_embeddings.npy',
    #     device='cuda:0'
    # )

