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
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider

#project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0, project_root)

# ====================================
# This is a example of loading the specified range of KV cache from ALIYUN
# ====================================

# ======= ALIYUN RAN CONTROL ==========
# ACCESS ID
# 
# ACCESS KEY
# 
# LOGIN PASSWORD
# 

auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())

# This is personal config of ALIYUN
endpoint = "https://oss-xx-xxxxxxxx.aliyuncs.com"
region = "xx-xxxxxxxx"

bucket = oss2.Bucket(auth, endpoint, "kvcache", region=region)

headers = {'x-oss-range-behavior': 'standard'}
object_name = 'kv_quant_0.pt'
object_stream = bucket.get_object(object_name, byte_range=(500, 2000), headers=headers)
print('standard get 500~2000 http status code:', object_stream.read())
