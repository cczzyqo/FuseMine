import torch
import esm
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

def get_protein_embeddings(csv_path, protein_column, save_path, batch_size,device):


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    

    device = torch.device(device)
    print(device)

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(device)
    

    df = pd.read_csv(csv_path)
    protein_sequences = df[protein_column].values

    all_embeddings = []
    

    for i in tqdm(range(0, len(protein_sequences), batch_size), desc="Processing proteins"):

        batch_sequences = protein_sequences[i:i + batch_size]
        

        data = [(f"protein_{j}", seq[:1022]) for j, seq in enumerate(batch_sequences)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        

        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]

            for j, tokens_len in enumerate(batch_lens):
                sequence_representation = token_representations[j, 1:tokens_len-1].mean(0)
                sequence_representation = sequence_representation.cpu().numpy()
                all_embeddings.append(sequence_representation)
    

    embeddings_array = np.array(all_embeddings, dtype = np.float64)
    

    np.save(save_path, embeddings_array)

    
    return embeddings_array



  
        
if __name__ == "__main__":
    csv_path = "protein.csv"
    protein_column = "protein"
    
    embeddings = get_protein_embeddings(
        csv_path=csv_path,
        protein_column=protein_column,
        batch_size=25,
        save_path='./data/protein_embeddings.npy',
        device='cuda:0'
    )
    
