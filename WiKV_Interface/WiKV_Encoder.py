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
        #self.bin_list = [20,20,18.18,14,12,10]
        self.bin_list = [24,22,20.18,16,14,10]
        self.layer_group = 6
        self.batch_size = 15000
        self.max_deviation = 6000
         
        
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
        """
        x = F.normalize(self.semantic_kv[max(0,(batch_id)*self.batch_size):min(self.kv_seq_len, (batch_id+1) * self.batch_size),0,:],p=2,dim=1)
        y = F.normalize(self.semantic_kv[max(0,(batch_id)*self.batch_size):min(self.kv_seq_len, (batch_id+1) * self.batch_size),0,:],p=2,dim=1)
        dist_matrix = x @ y.T

        x = F.normalize(self.semantic_kv[max(0,(batch_id)*self.batch_size):min(self.kv_seq_len, (batch_id+1) * self.batch_size),1,:],p=2,dim=1)
        y = F.normalize(self.semantic_kv[max(0,(batch_id)*self.batch_size):min(self.kv_seq_len, (batch_id+1) * self.batch_size),1,:],p=2,dim=1)
        dist_matrix1 = x @ y.T
    
        # normalize the dist_matrix to [0,1]
        dist_matrix = 1 - (dist_matrix + 1) / 2
        dist_matrix1 = 1 - (dist_matrix1 + 1) / 2
        self.dist_matrix = (dist_matrix + dist_matrix1) / 2
        """
        x = self.semantic_kv[max(0,(batch_id)*self.batch_size):min(self.kv_seq_len, (batch_id+1) * self.batch_size),1,:]
        y = self.semantic_kv[max(0,(batch_id)*self.batch_size):min(self.kv_seq_len, (batch_id+1) * self.batch_size),0,:]
        dist_matrix = torch.cdist(y,y,p=1) + torch.cdist(x,x,p=1)
        self.dist_matrix = dist_matrix

        return dist_matrix

    def PCA_sim_sort(self, batch_id):
        x = self.semantic_kv[
            max(0, batch_id * self.batch_size) : min(self.kv_seq_len, (batch_id + 1) * self.batch_size),
            0, :
        ].float()

        y = self.semantic_kv[
            max(0, batch_id * self.batch_size) : min(self.kv_seq_len, (batch_id + 1) * self.batch_size),
            1, :
        ].float()

        U, S, Vtx = torch.linalg.svd(x, full_matrices=False)
        U, S, Vty = torch.linalg.svd(y, full_matrices=False)

        pcx1 = Vtx[0]
        pcy1 = Vty[0]
        projectionx = x @ pcx1
        projectiony = y @ pcy1

        sorted_indicex = torch.argsort(projectionx, descending=True)
        sorted_indicex = [val.item() + batch_id * self.batch_size for val in sorted_indicex]

        sorted_indicey = torch.argsort(projectiony, descending=True)
        sorted_indicey = [val.item() + batch_id * self.batch_size for val in sorted_indicey]

        return sorted_indicex

    def greedy_sort(self, batch_id: int) -> list:
        """
        Use greedy nearest neighbor algorithm to generate an approximate shortest path permutation based on the distance matrix.
        
        Args:
            dist_matrix: (B, B) distance matrix (e.g., L1/L2 distance), symmetric, diagonal is 0
            
        Returns:
            sorted_indices: (B,) permutation indices so that adjacent points are as close as possible
        """
        B = self.dist_matrix.size(0)
        if B <= 1:
            return torch.arange(B, device=self.dist_matrix.device)

        # Initialization
        visited = torch.zeros(B, dtype=torch.bool, device=self.dist_matrix.device)
        path = torch.empty(B, dtype=torch.long, device=self.dist_matrix.device)

        # Start point is 0 (can be changed to random)
        current = 0
        path[0] = current
        visited[current] = True

        # Precompute node indices for masking
        node_indices = torch.arange(B, device=self.dist_matrix.device)

        # Greedily select nearest neighbor with ±500 constraint
        for i in range(1, B):
            distances = self.dist_matrix[current].clone()  # 避免 inplace 修改原矩阵

            # Mask out visited nodes
            distances = distances.masked_fill(visited, float('inf'))

            # Create mask for |idx - current| <= 500
            within_range = (node_indices - current).abs() <= self.max_deviation
            # Also ensure we don't go out of [0, B-1] — but abs already handles it

            # Combine: only allow unvisited AND within ±500
            valid_mask = visited.logical_not() & within_range
            if not valid_mask.any():
                valid_mask = visited.logical_not()

            # Mask out invalid nodes (those not in range)
            distances = distances.masked_fill(~valid_mask, float('inf'))

            next_node = torch.argmin(distances)
            path[i] = next_node  
            visited[next_node] = True
            current = next_node

        path = path + batch_id * self.batch_size

        return path

    def constrained_two_opt(self, max_iter=4, batch_id=0, improve_threshold=3):

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
            for i in range(num,n,random.randint(3, 5)):
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
                        best_solution = seq
                        best_distance = current_distance

                        #print(f"Iter {iteration}: swap ({i}, {j}), new dist = {best_distance:.2f}")

                        improved = True 


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
        
        # using cosine similarity as distance, we find 
        '''
        if not os.path.exists(f"{self.args.save_encode_dir}/Inflation"):
            os.makedirs(f"{self.args.save_encode_dir}/Inflation", exist_ok=True)
        total_batches = self.kv_seq_len // self.batch_size + 1
        for batch_id in range(total_batches):
            # if calculated, skip the batch
            if not os.path.exists(f"{self.args.save_encode_dir}/Inflation/seq_inflation_{session_id}_batch{batch_id}_.pt"):

                self.calculate_dist_matrix(batch_id=batch_id)
                solu, dist = self.constrained_two_opt(batch_id=batch_id)
                torch.save(solu, f"{self.args.save_encode_dir}/Inflation/seq_inflation_{session_id}_batch{batch_id}_.pt")
        '''
        # using greedy and 1 norm dist to sort the KV vectors
        if not os.path.exists(f"{self.args.save_encode_dir}/Inflation_greedy"):
            os.makedirs(f"{self.args.save_encode_dir}/Inflation_greedy", exist_ok=True)
        total_batches = self.kv_seq_len // self.batch_size + 1
        for batch_id in range(total_batches):
            # if calculated, skip the batch
            if not os.path.exists(f"{self.args.save_encode_dir}/Inflation_greedy/seq_inflation_{session_id}_batch{batch_id}_.pt"):
                self.calculate_dist_matrix(batch_id=batch_id)
                solu = self.greedy_sort(batch_id=batch_id)
                torch.save(solu, f"{self.args.save_encode_dir}/Inflation_greedy/seq_inflation_{session_id}_batch{batch_id}_.pt")
        
        

    def Inflation_Control(self, session_id):
        
        # ===============
        # modify the semantic kv cache based on the modified seq
        # ==============

        # check if the inflation control sequence is caculated
        if not os.path.exists(f"{self.args.save_encode_dir}/Inflation_greedy"):
            print("Error in loading inflation control seq, please calculate first with inflation_seq...")
            sys.exit(1)

        modify_seq = []
        for batch_id in range(self.kv_seq_len // self.batch_size + 1):
            tmp = torch.load(f"{self.args.save_encode_dir}/Inflation_greedy/seq_inflation_{session_id}_batch{batch_id}_.pt")
            modify_seq.extend(tmp)

        modified_sequence = self.sorted_sequence[modify_seq]

        k_seq = self.kv_quant[modified_sequence[:,0],0,modified_sequence[:,1],modified_sequence[:,2],:]
        v_seq = self.kv_quant[modified_sequence[:,0],1,modified_sequence[:,1],modified_sequence[:,2],:]

        # k_seq = self.kv_quant[self.sorted_sequence[:,0],0,self.sorted_sequence[:,1],self.sorted_sequence[:,2],:]
        # v_seq = self.kv_quant[self.sorted_sequence[:,0],1,self.sorted_sequence[:,1],self.sorted_sequence[:,2],:]

        #k_seq = self.kv_quant[:,0,:,:,:].reshape(-1,self.kv_quant.size(-1))
        #v_seq = self.kv_quant[:,1,:,:,:].reshape(-1,self.kv_quant.size(-1))

        if not os.path.exists(f"{self.args.save_encode_dir}Huffman"):
            os.makedirs(f"{self.args.save_encode_dir}Huffman", exist_ok=True)

        code_size = 0

        '''
        # Huffman Tree construction， we construct book for each batch
        CODE_SIZE = 1_000 * 128

        code_final = []
        code_size = 0
        flat_deltas, first_sample = delta_encode(k_seq)
        flat_deltas = flat_deltas.tolist()
        huff = HuffmanCodec()
        
        for i in range(0,len(flat_deltas), CODE_SIZE):
            codebook_path = f"{self.args.save_encode_dir}Huffman/codebook_key_{session_id}_{i}.pt"
            #if not os.path.exists(codebook_path):
            #    os.makedirs(f"{self.args.save_encode_dir}Huffman", exist_ok=True)
            huff.build_codebook(flat_deltas[i:i + CODE_SIZE])
            huff.save_codebook(codebook_path)
        #else:
        #    huff.load_codebook(codebook_path)

        def encode_key_chunk(args):
            chunk, i = args
            codefile_path = f"{self.args.save_encode_dir}Huffman/codebook_key_{session_id}_{i}.pt"
            huff = HuffmanCodec()
            huff.load_codebook(codefile_path)

            return huff.encode(chunk)

        # Huffman encoding
        # Use paraller to make it more efficient
        
        print(f"HUffman encode chunk size: {len(flat_deltas)}")
        chunks = [flat_deltas[i:i + CODE_SIZE] for i in range(0, len(flat_deltas), CODE_SIZE)]

        args = [(chunk, i * CODE_SIZE) for i, chunk in enumerate(chunks)]
        with ThreadPoolExecutor(max_workers=16) as executor:
            encoded_chunks = list(executor.map(encode_key_chunk, args))

        code = ''.join(encoded_chunks)
        code_byte = bits_to_bytes(code)
        code_final += code_byte
        code_size += len(code_byte)/1024/1024
        print(f"The code size of key: {len(code)/8/1024/1024:.1f}MB")


        flat_deltas, first_sample = delta_encode(v_seq)
        flat_deltas = flat_deltas.tolist()
        huff = HuffmanCodec()
        
        for i in range(0,len(flat_deltas), CODE_SIZE):
            codebook_path = f"{self.args.save_encode_dir}Huffman/codebook_val_{session_id}_{i}.pt"
            #if not os.path.exists(codebook_path):
            #    os.makedirs(f"{self.args.save_encode_dir}Huffman", exist_ok=True)
            huff.build_codebook(flat_deltas[i:i + CODE_SIZE])
            huff.save_codebook(codebook_path)

        def encode_val_chunk(args):
            chunk, i = args
            codefile_path = f"{self.args.save_encode_dir}Huffman/codebook_val_{session_id}_{i}.pt"
            huff = HuffmanCodec()
            huff.load_codebook(codefile_path)

            return huff.encode(chunk)

        chunks = [flat_deltas[i:i + CODE_SIZE] for i in range(0, len(flat_deltas), CODE_SIZE)]
        args = [(chunk, i * CODE_SIZE) for i, chunk in enumerate(chunks)]
        with ThreadPoolExecutor(max_workers=16) as executor:
            encoded_chunks = list(executor.map(encode_val_chunk, args))

        code = ''.join(encoded_chunks)
        code_byte = bits_to_bytes(code)
        print(f"The code size of value: {len(code)/8/1024/1024:.1f}MB")
        code_final += code_byte
        code_size += len(code_byte)/1024/1024
        # torch.save(code_final, f"{self.args.save_encode_dir}Huffman/code_final_{session_id}.pt")
        '''
        
        return modified_sequence, code_size