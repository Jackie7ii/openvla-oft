import tensorflow_datasets as tfds
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os
from tqdm import tqdm



def load_rlds_episodes(data_root, task_suite_name):
    cache_dir = os.path.join(data_root, f"{task_suite_name}_cache")

    if os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0:
        print(f"[data] loading from cache: {cache_dir}")
        episodes = []
        for fname in tqdm(sorted(os.listdir(cache_dir)), desc='[data] loading cache'):
            data = np.load(os.path.join(cache_dir, fname))
            episodes.append({k: data[k] for k in ['images', 'wrist_images', 'qpos', 'actions']})
        print(f"[data] loaded {len(episodes)} episodes from cache")
        return episodes

    dataset = tfds.load(task_suite_name, data_dir=data_root, split='train', download=False)
    print("[data] tfds.load done, starting episode iteration...")
    os.makedirs(cache_dir, exist_ok=True)

    episodes = []
    for i, episode in enumerate(dataset):
        if i % 10 == 0:
            print(f"[data] processing episode {i}...")

        steps = episode['steps']
        images, wrist_images, qpos, actions = [], [], [], []
        for step in steps:
            images.append(step['observation']['image'].numpy())
            wrist_images.append(step['observation']['wrist_image'].numpy())
            qpos.append(step['observation']['state'].numpy())
            actions.append(step['action'].numpy())

        ep = {
            'images': np.stack(images),
            'wrist_images': np.stack(wrist_images),
            'qpos': np.stack(qpos),
            'actions': np.stack(actions),
        }
        episodes.append(ep)
        np.savez(
            os.path.join(cache_dir, f"episode_{i:04d}.npz"),
            images=ep['images'],
            wrist_images=ep['wrist_images'],
            qpos=ep['qpos'],
            actions=ep['actions'],
        )

    print(f"[data] saved {len(episodes)} episodes to cache: {cache_dir}")
    return episodes

def get_norm_stats(episodes):
    all_actions = np.concatenate([ep['actions'] for ep in episodes], axis=0)
    action_mean = all_actions.mean(axis=0)
    action_std = np.clip(all_actions.std(axis=0), a_min=1e-2, a_max= np.inf)

    all_states = np.concatenate([ep['qpos'] for ep in episodes], axis=0)
    state_mean = all_states.mean(axis=0)
    state_std = np.clip(all_states.std(axis=0), a_min=1e-2, a_max=np.inf)

    return {
        'action_mean': action_mean,
        'action_std': action_std,
        'state_mean': state_mean,
        'state_std': state_std
    }

class LIBEROEpisodeDataset(Dataset):
    def __init__(self, episodes, camera_names, num_queries, norm_stats):
        super().__init__()
        self.episodes = episodes
        self.camera_names = camera_names
        self.num_queries = num_queries
        self.norm_stats = norm_stats

        self.action_dim = episodes[0]['actions'].shape[1]

        # use all steps
        # lengths = np.array([len(ep['actions']) for ep in episodes])
        # self.episodes_starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]) #lengths of episodes, used for indexing
        # self.total_steps = int(np.sum(lengths))

    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, episode_idx):
        # episode_idx = int(np.searchsorted(self.episodes_starts, idx, side='right') - 1) # find which episode this idx belongs to
        ep = self.episodes[episode_idx]
        # t = idx - self.episodes_starts[episode_idx]
        T = len(ep['actions'])
        t = np.random.choice(T)

        qpos = ep['qpos'][t]
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(ep[cam_name][t]) # TODO wrist images
        all_cam_images = np.stack(all_cam_images, axis=0) # concat on channel dimension, (H, W, 3*num_cam)
        # channel last → channel first
        image_data = np.transpose(all_cam_images, (0, 3, 1, 2))  # [num_cam, 3, 256, 256]
        # uint8 [0,255] → float32 [0,1]
        image_data = image_data.astype(np.float32) / 255.0

        # T = len(ep['actions'])
        action_chunk = np.zeros((self.num_queries, self.action_dim), dtype=np.float32)
        is_pad = np.zeros(self.num_queries, dtype=bool)

        actual_action_len = min(self.num_queries, T - t)
        action_chunk[:actual_action_len] = ep['actions'][t:t+actual_action_len]
        is_pad[actual_action_len:] = True

        #normalization
        action_chunk = (action_chunk - self.norm_stats['action_mean']) / self.norm_stats['action_std']
        qpos = (qpos - self.norm_stats['state_mean']) / self.norm_stats['state_std']

        image_data = torch.from_numpy(image_data)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(action_chunk).float()
        is_pad = torch.from_numpy(is_pad)

        return qpos_data, image_data, action_data, is_pad

def load_data(data_root, task_suite_name, camera_names, num_queries, batch_size_train, batch_size_eval):
    episodes = load_rlds_episodes(data_root, task_suite_name)
    num_episodes = len(episodes)
    indices = np.random.permutation(num_episodes) #suffle episode indices

    split = int(num_episodes * 0.8)
    train_episodes = [episodes[i] for i in indices[:split]]
    eval_episodes = [episodes[i] for i in indices[split:]]

    norm_stats = get_norm_stats(train_episodes)

    train_dataset = LIBEROEpisodeDataset(train_episodes, camera_names, num_queries, norm_stats)
    eval_dataset = LIBEROEpisodeDataset(eval_episodes, camera_names, num_queries, norm_stats)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4, prefetch_factor=1, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size_eval, shuffle=False, num_workers=4, prefetch_factor=1, pin_memory=True)

    return train_dataloader, eval_dataloader, norm_stats