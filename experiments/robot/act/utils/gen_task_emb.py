from sentence_transformers import SentenceTransformer
import numpy as np
from libero.libero import benchmark

model = SentenceTransformer('all-MiniLM-L6-v2')

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict['libero_goal']()

task_emb_dict = {}
for i in range(task_suite.n_tasks):
    desc = task_suite.get_task(i).language
    task_emb_dict[desc] = model.encode(desc)
    print(f'{i}: {desc}')

import os
save_dir = '/local_data/jl17265/projects/openvla-oft/experiments/robot/act/task_embeddings'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'libero_goal_task_embeddings.npy')
np.save(save_path, task_emb_dict)
print(f'saved to {save_path}')
