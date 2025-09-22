import pickle
import threading
import time
import torch
import os


# WiKV semantic encoder & inflation control

class WiKV_Encode:

    def __init__(self, args, seq_len, config, session, device='cpu'):
        self.args = args    
        self.seq_len = seq_len
        self.config = config
        self.impor_score = []
        self.session = session
        
    def Att_Loading(self):

        for i in range(self.config.num_hidden_layers):
            file_path = os.path.join(self.args.save_att_dir, f"attn_s{self.session}_l{i}.pt")
            attn_weights = torch.load(file_path)
            impor_scores = torch.sum(attn_weights, dim=1)
            impor_scores = impor_scores[:,0:self.seq_len]
            impor_scores = impor_scores.unsqueeze(0)
            self.impor_score.append(impor_scores)
        
        self.impor_score = torch.cat(self.impor_score, dim=0)
        print(self.impor_score.shape)
        
        
    def Semantic_Encode(self):
        flat_tensor = self.impor_score.flatten()
        sorted_indices_flat = torch.argsort(flat_tensor, descending=True)
        indices = torch.unravel_index(sorted_indices_flat, self.impor_score.shape)
        indices = torch.stack(indices, dim=1)
        print("排序后的三维索引数组形状:", indices.shape)
        print("前10个最大值的索引:")
        print(indices[:10])
        self.sorted_sequence = indices

        # re_order the KV cache
        file_path = os.path.join(self.args.save_kv_dir, f"raw_kv_{self.session}.pt")
        kv = torch.load(file_path)
        print(f"KV cache size is : {kv.shape}")
    # def 