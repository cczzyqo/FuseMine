import torch
import torch.nn as nn
from model.MOFA import *
from utils import to_3d,to_4d
# drug protein hiddendim=128
class LinearTransform_bert(nn.Module):
    def __init__(self):
        super(LinearTransform_bert, self).__init__()
        self.linear1 = nn.Linear(384, 512)#for seed 12 for better score it was on 384>64>128
        self.linear2 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        # Reshape the input tensor to [batch_size, 1024]
        x = x.view(x.size(0), -1)

        # Apply the first linear layer
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = x.unsqueeze(1)  # Add a singleton dimension to match [batch_size, 1, 512]

        # Apply the second linear layer
        x = torch.relu(self.linear2(x))

        return x

class LinearTransform_esm(nn.Module):
    def __init__(self):
        super(LinearTransform_esm, self).__init__()
        self.linear1 = nn.Linear(1280, 512)
        self.linear2 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        # Reshape the input tensor to [batch_size, 1024]
        x = x.view(x.size(0), -1)

        # Apply the first linear layer
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = x.unsqueeze(1)  # Add a singleton dimension to match [batch_size, 1, 512]

        # Apply the second linear layer
        x = torch.relu(self.linear2(x))

        return x


class molFusion(nn.Module):
    def __init__(self):
        super(molFusion, self).__init__()
        self.mofa =  MFeatureOrchestrationAggregation(128)

MFeatureOrchestrationAggregation
    def forward(self, A, B):
        result_1 = torch.matmul(A, B.transpose(1,2)) # [B, N, D] [B, D, 1] -> [B, N, 1]


        result_2 = torch.matmul(result_1, B) # [B, N, 1] [B, 1, D] -> [B, N, D]
        result_2 = to_3d(self.mofa(to_4d(result_2, result_2.size(1), 1)))

        final_result = torch.add(result_2, A) # [B, N, D] + [B, N, D] -> [B, N, D]
        return final_result


class proFusion(nn.Module):
    def __init__(self):
        super(proFusion, self).__init__()
        self.moga =  MFeatureOrchestrationAggregation(128)

    def forward(self, A, B):

        result_1 = torch.matmul(A, B.transpose(1,2))

        result_2 = torch.matmul(result_1, B)
        result_2 = to_3d(self.mofa(to_4d(result_2, result_2.size(1), 1)))
        final_result = torch.add(result_2, A)
        return final_result
        