import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dgl
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch.utils.data as data
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein

class Mydataset(Dataset):
    def __init__(self,samples,compounds,proteins,unique_smiles,unique_proteins,max_drug_nodes, max_protein_length):
        # samples 
        # ,smiles,sequence,interactions
        # 0,2142,81,1
        self.samples=samples
        self.len=len(samples)
        # llm embedding
        self.compounds = compounds
        self.proteins = proteins
        # pd dataframe
        self.smiles = unique_smiles
        self.sequences = unique_proteins
        
        self.max_drug_nodes = max_drug_nodes
        self.max_protein_length = max_protein_length

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
    def __getitem__(self, idx):
        #return self.compound[idx],self.adj[idx],self.protein[idx],self.correct_interaction[idx]
        one_sample = self.samples[idx]

        smile = self.smiles.iloc[one_sample[1]]['smiles']

        v_d = self.fc(smiles=smile, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()

        seq = self.sequences.iloc[one_sample[2]]['protein']
        v_p = integer_label_protein(seq, self.max_protein_length)

        drug_emd, protein_emd,y = self.compounds[int(one_sample[1])],self.proteins[int(one_sample[2])],torch.tensor([self.samples[idx,3]])
        return v_d, v_p,drug_emd, protein_emd,y,one_sample
    def __len__(self):
        return self.len


def load_tensor(file_name, dtype):
    data = np.load(file_name + '.npy', allow_pickle=True)
    
    # Assuming each element in data is an ndarray, convert each to the desired dtype and then to a tensor
    tensor_data = [torch.tensor(np.array(d, dtype=dtype), dtype=torch.float32) for d in data]
    
    return tensor_data

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2



def preparedata(type,dataset):
    dir_input = ('data/'+dataset+'/')
    # llm embedding
    compounds = load_tensor(dir_input + 'smiles_embeddings', np.float64)
    proteins = load_tensor(dir_input + 'protein_embeddings', np.float64)
    # samples 
    trainfiles = pd.read_csv(dir_input + type + '/train/' + 'samples.csv')
    validfiles = pd.read_csv(dir_input + type + '/valid/' + 'samples.csv')
    testfiles = pd.read_csv(dir_input + type + '/test/' + 'samples.csv')
    # string list
    unique_smiles = pd.read_csv(dir_input + 'smiles.csv')
    unique_proteins = pd.read_csv(dir_input + 'protein.csv')
    

    return trainfiles.values,validfiles.values,testfiles.values,compounds,proteins,unique_smiles,unique_proteins


def collatef(x):
    d, p, smile, esm, y, samples= zip(*x)
    d = dgl.batch(d)
    return d, torch.tensor(np.array(p)), torch.tensor(np.array(smile)), torch.tensor(np.array(esm)), torch.tensor(y),samples

def preparedataset(batch_size,type,dataset,num_worker,max_drug_nodes, max_protein_length):
    trainsamples,validsamples,testsamples,compounds,proteins,unique_smiles,unique_proteins=preparedata(type,dataset)
    trainloader = DataLoader(Mydataset(trainsamples,compounds,proteins,unique_smiles,unique_proteins,max_drug_nodes, max_protein_length),shuffle=True,batch_size=batch_size,collate_fn=collatef, drop_last=False,pin_memory=True,num_workers=num_worker)
    validloader = DataLoader(Mydataset(validsamples, compounds, proteins,unique_smiles,unique_proteins,max_drug_nodes, max_protein_length), shuffle=False, batch_size=batch_size,
                            collate_fn=collatef, drop_last=False,pin_memory=True,num_workers=num_worker)
    testloader = DataLoader(Mydataset(testsamples, compounds, proteins,unique_smiles,unique_proteins,max_drug_nodes, max_protein_length), shuffle=False, batch_size=batch_size,
                             collate_fn=collatef, drop_last=False, pin_memory=True,num_workers=num_worker)
    return trainloader,validloader,testloader,compounds,proteins

