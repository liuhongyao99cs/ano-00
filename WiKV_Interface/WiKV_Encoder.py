import pickle
import threading
import time
import torch
import os
import sys
import copy
import random
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

#project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0, project_root)

from src import *

# WiKV semantic encoder & inflation control

class WiKV_Encode:

    def __init__(self, args, seq_len, config, session, window_size, device='cpu'):

        # ============
        # args -> params, 
        # window_size -> layer dependency
        # session is the sample id
        # seq_len is the total token num of context
        # args is the parameter of dir and config
        # ============

        self.args = args    
        self.seq_len = seq_len
        self.config = config
        self.impor_score = []
        self.session = session
        self.window_size = window_size
        self.bin_list = [24,20,14,10,12]
        self.layer_group = 9
        self.batch_size = 10000
        self.max_deviation = 20
         
        
    def Att_Loading(self):

        # =======================
        # Load in attention score generated from Attention.py
        # Process the attention based on the later dependency
        # =====================

        print(f"WiKV load attention weights and process layer dependency...")
        for i in range(self.config.num_hidden_layers):
            file_path = os.path.join(self.args.save_att_dir, f"attn_s{self.session}_l{i}.pt")

            if not os.path.exists(file_path):
                print("Compute the attention weights first...")
                sys.exit(1)

            attn_weights = torch.load(file_path)
            impor_scores = torch.sum(attn_weights, dim=1) / attn_weights.shape[1]
            impor_scores = impor_scores[:,0:self.seq_len-1]
            impor_scores = impor_scores.unsqueeze(0)
            self.impor_score.append(impor_scores)
        
        self.impor_score = torch.cat(self.impor_score, dim=0)

        # handle the layer_dependency
        tensor = torch.zeros_like(self.impor_score)
        for j in range(self.config.num_hidden_layers):
            end = min(j + self.window_size, self.config.num_hidden_layers)
            tensor[j] = torch.sum(self.impor_score[j:end], dim=0)

        self.impor_score = tensor


        #print(self.impor_score)

        
        
    def Semantic_Encode(self):

        # =============================
        # First, sort the indices of attention weights -> semantic sequence
        # Second. layer-wise quantize
        # =============================

        print(f"WiKV semantic encoding begining")
        flat_tensor = self.impor_score.flatten()
        sorted_indices_flat = torch.argsort(flat_tensor, descending=True)
        indices = torch.unravel_index(sorted_indices_flat, self.impor_score.shape)
        indices = torch.stack(indices, dim=1).cpu()
        
        # print(f"Most important 10 KV vectors: {indices[:60]}")
        
        self.sorted_sequence = indices
        self.kv_seq_len = indices.shape[0]

        # re_order the KV cache
        file_path = os.path.join(self.args.save_kv_dir, f"raw_kv_{self.session}.pt")
        if not os.path.exists(file_path):
            print("Compute the KV cache for the session...")
            sys.exit(1)

        kv = torch.load(file_path)
        #print(f"KV cache size is : {kv.shape}")

        # layer-wise quantization
        kv_quant, max_q = layer_quantization(kv, self.bin_list, self.layer_group)
        kv_dequant = layer_dequantize(kv_quant, max_q, self.bin_list, self.layer_group)

        # semantic re-order
        #kv_quant_cpu = kv_quant.cpu()
        
        self.kv_quant = kv_quant
        self.semantic_kv = kv_quant[indices[:,0],:,indices[:,1],indices[:,2],:]

        return kv_quant, kv_dequant
    
    
    def calculate_dist_matrix(self, batch_id):

        # ================================
        # caclutate the dist between kv vectors in seq batch_id
        # ================================

        # calculate the num of KV vectors in the current batch
        lenx = 0
        if (batch_id + 1) * self.batch_size > self.kv_seq_len:
            lenx = self.kv_seq_len - (batch_id) * self.batch_size + 1
        else:
            lenx = self.batch_size


        initial_solution = torch.arange(0,lenx)
        dist_matrix = -torch.ones(len(initial_solution),len(initial_solution))
        #print(self.semantic_kv[1,0,:].unsqueeze(0))

        if self.semantic_kv is None:
            print("Error in performing semantic coding before inflation control...")
            sys.exit(1)


        self.semantic_kv = self.semantic_kv.float()

        # compute dist matrix based on semantic_kv
        x = F.normalize(self.semantic_kv[max(0,(batch_id)*self.batch_size):min(self.kv_seq_len, (batch_id+1) * self.batch_size),0,:],p=2,dim=1)
        y = F.normalize(self.semantic_kv[max(0,(batch_id)*self.batch_size):min(self.kv_seq_len, (batch_id+1) * self.batch_size),0,:],p=2,dim=1)
        dist_matrix = x @ y.T
    
        # normalize the dist_matrix to [0,1]
        dist_matrix = 1 - (dist_matrix + 1) / 2
        self.dist_matrix = dist_matrix

        return dist_matrix
    
    def constrained_two_opt(self, max_iter=3, batch_id=0, improve_threshold=0.01):

        # ========================
        # code inflation control to obtain a modified seq with higher efficiency
        # ========================

        print("Optimize the semantic sequence to control inflation...")

        n = 0
        if (batch_id+1) * self.batch_size > self.kv_seq_len:
            n = self.kv_seq_len - (batch_id) * self.batch_size 
        else:
            n = self.batch_size
        
        seq = list(range(n))  # initialize [0, 1, 2, ..., n-1]
        
        # calculate the distance between nodes in seq
        def calculate_total_distance(path):
            total = 0.0
            for i in range(1, n):
                total += self.dist_matrix[path[i-1], path[i]]
            return total

        current_distance = calculate_total_distance(seq)
        best_solution = copy.deepcopy(seq)
        best_distance = current_distance
        print(f"Initial distance: {current_distance:.2f}")

        # compute the distance delta of 
        def get_swap_delta(path, i, j):

            if i == j:
                return 0.0
            if i > j:
                i, j = j, i

            # maintain the values before swap
            delta = 0.0
            dist = self.dist_matrix
            val_i, val_j = path[i], path[j]  
           

            # --- delete old edge ---
            if i > 0:
                delta -= dist[path[i-1], val_i]
            if i < n - 1:
                delta -= dist[val_i, path[i+1]]
            if j > 0:
                delta -= dist[path[j-1], val_j]
            if j < n - 1:
                delta -= dist[val_j, path[j+1]]

            # if adjacent
            if j == i + 1:
                delta += dist[val_i, val_j]

            # --- add new edge ---
            if i > 0:
                delta += dist[path[i-1], val_j] 
            if i < n - 1:
                delta += dist[val_j, path[i+1]]
            if j > 0:
                delta += dist[path[j-1], val_i]  
            if j < n - 1:
                delta += dist[val_i, path[j+1]]

            if j == i + 1:
                delta -= dist[val_j, val_i]

            return delta

        # 
        def is_valid_swap(i, j, seq):
            return abs(seq[j] - i) <= self.max_deviation and abs(seq[i] - j) <= self.max_deviation
        

        for iteration in range(max_iter):
            improved = False

            # loop all swap pairs
            num = random.randint(0, 5)
            for i in range(num,n,random.randint(3, 7)):
                # select valid j under constraint
                st = max(seq[i]-self.max_deviation,i+2)
                ed = min(n,seq[i]+self.max_deviation)
                for j in range(st,ed,2): 
                    #if not is_valid_swap(i, j, seq):
                        #continue

                    delta = get_swap_delta(seq, i, j)

                    if delta < -improve_threshold:  

                        seq[i], seq[j] = seq[j], seq[i]
                        current_distance += delta
                        best_solution = copy.deepcopy(seq)
                        best_distance = current_distance

                        #print(f"Iter {iteration}: swap ({i}, {j}), new dist = {best_distance:.2f}")

                        improved = True 

                        break


            if not improved:
                break

        current_distance = calculate_total_distance(seq)
        print(f"Batch {batch_id}: Optimized seq distance: {current_distance:.2f}")
        best_solution = [ val + batch_id * self.batch_size for val in best_solution ]

        return best_solution, best_distance

    def Inflation_Seq(self, session_id):

        # =======================
        # if inflation_seq exists, read; otherwise, compute. Compute takes a lot of time
        # ======================

        if not os.path.exists(f"{self.args.save_encode_dir}/Inflation"):
            os.makedirs(f"{self.args.save_encode_dir}/Inflation", exist_ok=True)
        total_batches = self.kv_seq_len // self.batch_size + 1
        for batch_id in range(total_batches):
            # if calculated, skip the batch
            if not os.path.exists(f"{self.args.save_encode_dir}/Inflation/seq_inflation_{session_id}_batch{batch_id}_.pt"):

                self.calculate_dist_matrix(batch_id=batch_id)
                solu, dist = self.constrained_two_opt(batch_id=batch_id)
                torch.save(solu, f"{self.args.save_encode_dir}/Inflation/seq_inflation_{session_id}_batch{batch_id}_.pt")

    def Inflation_Control(self, session_id):
        
        # ===============
        # modify the semantic kv cache based on the modified seq
        # ==============

        # check if the inflation control sequence is caculated
        if not os.path.exists(f"{self.args.save_encode_dir}/Inflation"):
            print("Error in loading inflation control seq, please calculate first with inflation_seq...")
            sys.exit(1)

        modify_seq = []
        for batch_id in range(self.kv_seq_len // self.batch_size + 1):
            tmp = torch.load(f"{self.args.save_encode_dir}/Inflation/seq_inflation_{session_id}_batch{batch_id}_.pt")
            modify_seq.extend(tmp)

        modified_sequence = self.sorted_sequence[modify_seq]
        k_seq = self.kv_quant[modified_sequence[:,0],0,modified_sequence[:,1],modified_sequence[:,2],:]
        v_seq = self.kv_quant[modified_sequence[:,0],1,modified_sequence[:,1],modified_sequence[:,2],:]


        if not os.path.exists(f"{self.args.save_encode_dir}Huffman"):
            os.makedirs(f"{self.args.save_encode_dir}Huffman", exist_ok=True)

        # Huffman Tree construction
        code_final = []
        code_size = 0
        flat_deltas, first_sample = delta_encode(k_seq)
        flat_deltas = flat_deltas.tolist()
        huff = HuffmanCodec()
        codebook_path = f"{self.args.save_encode_dir}Huffman/codebook_key_{session_id}.pt"
        #if not os.path.exists(codebook_path):
        #    os.makedirs(f"{self.args.save_encode_dir}Huffman", exist_ok=True)
        huff.build_codebook(flat_deltas)
        huff.save_codebook(codebook_path)
        #else:
        #    huff.load_codebook(codebook_path)

        def encode_chunk(chunk):
            return huff.encode(chunk)

        # Huffman encoding
        # Use paraller to make it more efficient
        '''
        print(f"HUffman encode chunk size: {len(flat_deltas)}")
        CHUNK_SIZE = 8_000_000
        chunks = [flat_deltas[i:i + CHUNK_SIZE] for i in range(0, len(flat_deltas), CHUNK_SIZE)]

        with ThreadPoolExecutor(max_workers=16) as executor: 
            encoded_chunks = list(executor.map(encode_chunk, chunks))

        code = ''.join(encoded_chunks)
        code_byte = bits_to_bytes(code)
        code_final += code_byte
        code_size += len(code_byte)/1024/1024
        print(f"The code size of key: {len(code_byte)/1024/1024:.1f}MB")

        flat_deltas, first_sample = delta_encode(v_seq)
        flat_deltas = flat_deltas.tolist()
        huff = HuffmanCodec()
        #if not os.path.exists(f"{self.args.save_encode_dir}Huffman/codebook_val_{session_id}.pt"):
        #    os.makedirs(f"{self.args.save_encode_dir}Huffman", exist_ok=True)
        huff.build_codebook(flat_deltas)
        huff.save_codebook(f"{self.args.save_encode_dir}Huffman/codebook_val_{session_id}.pt")
        #else:
        #    huff.load_codebook(f"{self.args.save_encode_dir}Huffman/codebook_val_{session_id}.pt")


        CHUNK_SIZE = 8_000_000
        chunks = [flat_deltas[i:i + CHUNK_SIZE] for i in range(0, len(flat_deltas), CHUNK_SIZE)]

        with ThreadPoolExecutor(max_workers=16) as executor: 
            encoded_chunks = list(executor.map(encode_chunk, chunks))

        code = ''.join(encoded_chunks)
        code_byte = bits_to_bytes(code)
        print(f"The code size of value: {len(code_byte)/1024/1024:.1f}MB")
        code_final += code_byte
        code_size += len(code_byte)/1024/1024
        # torch.save(code_final, f"{self.args.save_encode_dir}Huffman/code_final_{session_id}.pt")
        '''
        return modified_sequence, code_size