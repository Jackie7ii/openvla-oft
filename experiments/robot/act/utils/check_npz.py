import numpy as np
import os

cache_dir = '/local_data/jl17265/projects/openvla-oft/modified_libero_rlds/libero_spatial_no_noops_cache'

files = sorted(os.listdir(cache_dir))
print(f'Total episodes: {len(files)}')

# check the n th episode
data = np.load(os.path.join(cache_dir, files[0]))
print(f'\nEpisode 0: {files[0]}')
for k in data.files:
    print(f'  {k}: shape={data[k].shape}, dtype={data[k].dtype}')