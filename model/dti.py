import torch
from torch import nn
from dgllife.model.gnn import GCN
import torch.nn.functional as F
import math
import pytorch_lightning as pl
from utils import to_3d, to_4d, calculate_metrics
import numpy as np
from model.modules import LinearTransform_bert, LinearTransform_esm, molFusion, proFusion
from model.RDFM import SRADiffAttn
class DTIModel(pl.LightningModule):
    def __init__(self, configs, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.configs = configs
        self.learning_rate = learning_rate
        
        # Model components
        self.drug_extractor = MoleculeGCN(configs)
        self.prot_extractor = ProteinCNN(configs)

        # fusion cnn/gnn llms
        self.molecule_bert = LinearTransform_bert()
        self.protein_esm = LinearTransform_esm()
        self.mol_fusion = molFusion()
        self.pro_fusion = proFusion()

        
        self.sdattn_d_layers = nn.ModuleList([
            SRADiffAttn(
                embed_dim=configs.SDATTN.Embed_Dim,
                depth=i,  
                num_heads=configs.SDATTN.Num_Heads,
                num_kv_heads=configs.SDATTN.Num_KV_Heads,
                sr_ratio=configs.SDATTN.SR_Ratio
            ) for i in range(configs.SDATTN.Depth)
        ])
        
        self.sdattn_p_layers = nn.ModuleList([
            SRADiffAttn(
                embed_dim=configs.SDATTN.Embed_Dim,
                depth=i,  
                num_heads=configs.SDATTN.Num_Heads,
                num_kv_heads=configs.SDATTN.Num_KV_Heads,
                sr_ratio=configs.SDATTN.SR_Ratio
            ) for i in range(configs.SDATTN.Depth)
        ])
        self.fusion = MBCA(configs)
        self.mlp_classifier = DropoutMLP(configs)
        
        # Loss function 
        self.criterion = nn.CrossEntropyLoss()
        
        self.validation_step_outputs = []
        

    def forward(self, d_graph, p_feat, d_emb, p_emb, mode='train'):
        v_d = self.drug_extractor(d_graph)
        v_p = self.prot_extractor(p_feat)
        v_emb = self.molecule_bert(d_emb)
        v_p_emb = self.protein_esm(p_emb)


        v_d = self.mol_fusion(v_d, v_emb)
        v_p = self.pro_fusion(v_p, v_p_emb)


        residual_d = v_d
        residual_p = v_p
        
        for i, sdattn in enumerate(self.sdattn_d_layers):
            v_d = sdattn(v_d)
        v_d = v_d + residual_d
        for i, sdattn in enumerate(self.sdattn_p_layers):
            v_p = sdattn(v_p)
        v_p = v_p + residual_p


        f, attn = self.fusion(v_d, v_p)
        score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, attn, score

    def training_step(self, batch, batch_idx):
        d_graph, p_feat, d_emb, p_emb, labels, sample= batch
        _, _, _, scores = self(d_graph, p_feat, d_emb, p_emb, mode='train')
        loss = self.criterion(scores, labels.long())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        d_graph, p_feat, d_emb, p_emb, labels, sample= batch
        _, _, _, scores = self(d_graph, p_feat, d_emb, p_emb, mode='eval')
        loss = self.criterion(scores, labels.long())
        
        y_scores = F.softmax(scores, dim=1).cpu().detach().numpy()
        y_preds = np.argmax(y_scores, axis=1)
        y_scores_positive = y_scores[:, 1]  
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        result = {
            'val_loss': loss, 
            'y_scores': y_scores_positive, 
            'y_preds': y_preds, 
            'labels': labels.cpu().numpy()
        }
        self.validation_step_outputs.append(result)
        
        return result

        
    def on_validation_epoch_end(self):

        y_true = np.concatenate([x['labels'] for x in self.validation_step_outputs])
        y_pred = np.concatenate([x['y_preds'] for x in self.validation_step_outputs])
        y_score = np.concatenate([x['y_scores'] for x in self.validation_step_outputs])
        
        metrics = calculate_metrics(y_true, y_pred, y_score)
        
        self.log('val_auroc', metrics['auroc'], on_step=False, on_epoch=True, prog_bar=True)
        
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        d_graph, p_feat, d_emb, p_emb, labels, sample= batch
        _, _, _, scores = self(d_graph, p_feat, d_emb, p_emb, mode='eval')
        loss = self.criterion(scores, labels.long())
        
        y_scores = F.softmax(scores, dim=1).cpu().detach().numpy()
        y_preds = np.argmax(y_scores, axis=1)
        y_scores_positive = y_scores[:, 1]  
        
        self.log('test_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        
        return {
            'test_loss': loss, 
            'y_scores': y_scores_positive, 
            'y_preds': y_preds, 
            'labels': labels.cpu()
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# Original model components remain the same
class MoleculeGCN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.in_feat = configs.Drug.Node_In_Feat
        self.dim_embedding = configs.Drug.Node_In_Embedding
        self.hidden_feats = configs.Drug.Hidden_Layers
        self.padding = configs.Drug.Padding
        self.activation = configs.Drug.GCN_Activation

        self.init_linear = nn.Linear(self.in_feat, self.dim_embedding, bias=False)
        if self.padding:
            with torch.no_grad():
                self.init_linear.weight[-1].fill_(0)
        self.gcn = GCN(in_feats=self.dim_embedding, hidden_feats=self.hidden_feats, activation=self.activation)
        self.output_feats = self.hidden_feats[-1]

    def forward(self, batch_d_graph):
        node_feats = batch_d_graph.ndata['h']
        node_feats = self.init_linear(node_feats)
        node_feats = self.gcn(batch_d_graph, node_feats)
        batch_size = batch_d_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class ProteinCNN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.embedding_dim = configs.Protein.Embedding_Dim
        self.num_filters = configs.Protein.Num_Filters
        self.kernel_size = configs.Protein.Kernel_Size
        self.padding = configs.Protein.Padding

        if self.padding:
            self.embedding = nn.Embedding(26, self.embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, self.embedding_dim)
        in_out_ch = [self.embedding_dim] + self.num_filters
        kernels = self.kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_out_ch[0], out_channels=in_out_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_out_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_out_ch[1], out_channels=in_out_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_out_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_out_ch[2], out_channels=in_out_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_out_ch[3])

    def forward(self, p_feat):
        p_feat = self.embedding(p_feat.long())
        p_feat = p_feat.transpose(2, 1)
        p_feat = F.relu(self.bn1(self.conv1(p_feat)))
        p_feat = F.relu(self.bn2(self.conv2(p_feat)))
        p_feat = F.relu(self.bn3(self.conv3(p_feat)))
        p_feat = p_feat.transpose(2, 1)
        return p_feat


def reconstruct(x_1, x_2):
    x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
    x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
    return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)




class MBCA(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.positional_drug = PositionalEncoding(configs.MBCA.Hidden_Size, max_len=configs.Drug.Max_Nodes)
        self.positional_prot = PositionalEncoding(configs.MBCA.Hidden_Size, max_len=configs.Protein.After_CNN_Length)
        self.attn_map = AttenMapNHeads(configs)
        self.attention_fc_dp = nn.Linear(configs.MBCA.Num_Heads, configs.MBCA.Hidden_Size)
        self.attention_fc_pd = nn.Linear(configs.MBCA.Num_Heads, configs.MBCA.Hidden_Size)

    def forward(self, drug, protein):
        drug = self.positional_drug(drug)
        protein = self.positional_prot(protein)
        attn_map = self.attn_map(drug, protein)
        att_dp = F.softmax(attn_map, dim=-1)
        att_pd = F.softmax(attn_map, dim=-2)
        attn_matrix = 0.5 * att_dp + 0.5 * att_pd
        drug_attn = self.attention_fc_dp(torch.mean(attn_matrix, -1).transpose(-1, -2))
        protein_attn = self.attention_fc_pd(torch.mean(attn_matrix, -2).transpose(-1, -2))
        drug_attn = F.sigmoid(drug_attn)
        protein_attn = F.sigmoid(protein_attn)
        drug = drug + drug * drug_attn
        protein = protein + protein * protein_attn
        drug, _ = torch.max(drug, 1)
        protein, _ = torch.max(protein, 1)
        pair = torch.cat([drug, protein], dim=1)
        return pair, (drug_attn, protein_attn)


class DropoutMLP(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(configs.MLP.In_Dim * 2, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, configs.MLP.Binary)

    def forward(self, pair):
        pair = self.dropout1(pair)
        fully1 = F.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = F.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = F.leaky_relu(self.fc3(fully2))
        pred = self.out(fully3)
        return pred


class AttenMapNHeads(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.hid_dim = configs.MBCA.Hidden_Size
        self.n_heads = configs.MBCA.Num_Heads
        assert self.hid_dim % self.n_heads == 0
        self.f_q = nn.Linear(self.hid_dim, self.hid_dim)
        self.f_k = nn.Linear(self.hid_dim, self.hid_dim)
        self.d_k = self.hid_dim // self.n_heads

    def forward(self, d, p):
        batch_size = d.shape[0]
        Q = self.f_q(d)
        K = self.f_k(p)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        return attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


