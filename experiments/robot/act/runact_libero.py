import os
import sys
import argparse
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
from collections import deque
from libero.libero import benchmark

import wandb

sys.path.append("../..")
from experiments.robot.act.policy import ACTPolicy
from experiments.robot.act.datasetdeal import load_data
from experiments.robot.act.config import ACT_LIBERO_CONFIG, TASK_MAX_STEPS
from experiments.robot.libero.libero_utils import (
    get_libero_env,
    get_libero_dummy_action,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)

def make_policy(args):
    policy = ACTPolicy(args)
    return policy

def forward_pass(data, policy, device):
    qpos_data, image_data, action_data, is_pad, task_emb = data
    qpos_data   = qpos_data.to(device)
    image_data  = image_data.to(device)
    action_data = action_data.to(device)
    is_pad      = is_pad.to(device)
    task_emb = task_emb.to(device)
    return policy(qpos_data, image_data, action_data, is_pad, task_emb)

def get_obs_tensors(obs, norm_stats, camera_names, device):
    image       = get_libero_image(obs)
    wrist_image = get_libero_wrist_image(obs)
    cam_map     = {'images': image, 'wrist_images': wrist_image}
    all_cam_images = np.stack([cam_map[name] for name in camera_names], axis=0)
    all_cam_images = np.transpose(all_cam_images, (0, 3, 1, 2)).astype(np.float32) / 255.0
    image_tensor   = torch.from_numpy(all_cam_images).unsqueeze(0).to(device)

    qpos = np.concatenate([
        obs['robot0_eef_pos'],
        quat2axisangle(obs['robot0_eef_quat']),
        obs['robot0_gripper_qpos'],
    ])
    qpos = (qpos - norm_stats['state_mean']) / norm_stats['state_std']
    qpos_tensor = torch.from_numpy(qpos).float().unsqueeze(0).to(device)

    return qpos_tensor, image_tensor


def rollout_eval(policy, norm_stats, args, device):
    """训练中调用的快速 rollout eval，用 num_trials_per_task 控制每个 task 的试验次数。"""
    policy.eval()

    
    eval_seed = args.get('eval_seed', 42)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)

    post_process   = lambda a: a * norm_stats['action_std'] + norm_stats['action_mean']
    suite_key      = args['task_suite_name'].replace('_no_noops', '')
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict[suite_key]()
    num_tasks      = task_suite.n_tasks

    num_trials     = args.get('num_trials_per_task', 5)
    num_steps_wait = args.get('num_steps_wait', 10)
    num_open_loop  = args.get('num_open_loop_steps', 8)  
    temporal_agg   = args.get('temporal_agg', False)    
    max_steps      = TASK_MAX_STEPS.get(suite_key, 300)
    num_queries    = args['num_queries']
    task_emb_dict = np.load(args['task_emb_path'], allow_pickle=True).item()

    total_episodes, total_successes = 0, 0
    for task_id in range(num_tasks):
        task           = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_desc = get_libero_env(task, 'act', resolution=256)
        task_emb = torch.from_numpy(task_emb_dict[task_desc]).float().unsqueeze(0).to(device)

        task_episodes, task_successes = 0, 0
        for trial_idx in range(num_trials):
            initial_state = initial_states[trial_idx % len(initial_states)]
            env.reset()
            obs     = env.set_init_state(initial_state)
            t       = 0
            success = False
            action_queue = deque()

            if temporal_agg:
                all_time_actions = np.zeros((max_steps, max_steps + num_queries, 7))

            while t < max_steps + num_steps_wait:
                if t < num_steps_wait:
                    obs, _, _, _ = env.step(get_libero_dummy_action('act'))
                    t += 1
                    continue

                t_rel = t - num_steps_wait
                if temporal_agg:
                    qpos_tensor, image_tensor = get_obs_tensors(
                        obs, norm_stats, args['camera_names'], device)
                    with torch.inference_mode():
                        action_chunk = policy(qpos_tensor, image_tensor, task_emb = task_emb)
                    action_chunk = action_chunk.squeeze(0).cpu().numpy()
                    action_chunk = post_process(action_chunk)
                    fill_len = min(num_queries, max_steps - t_rel)
                    all_time_actions[t_rel, t_rel:t_rel + fill_len] = action_chunk[:fill_len]
                    actions_for_curr = all_time_actions[:t_rel + 1, t_rel]
                    k = args.get('agg_k', 0.01)
                    weights = np.exp(-k * np.arange(len(actions_for_curr))[::-1])
                    weights /= weights.sum()
                    action = (actions_for_curr * weights[:, None]).sum(axis=0)
                else:
                    if len(action_queue) == 0:
                        qpos_tensor, image_tensor = get_obs_tensors(
                            obs, norm_stats, args['camera_names'], device)
                        with torch.inference_mode():
                            action_chunk = policy(qpos_tensor, image_tensor, task_emb = task_emb)
                        action_chunk = action_chunk.squeeze(0).cpu().numpy()
                        action_chunk = post_process(action_chunk)
                        for i in range(min(num_open_loop, len(action_chunk))):
                            action_queue.append(action_chunk[i])
                    action = action_queue.popleft()

                obs, _, done, _ = env.step(action.tolist())
                if done:
                    success = True
                    break
                t += 1

            task_episodes  += 1
            total_episodes += 1
            if success:
                task_successes  += 1
                total_successes += 1

        task_rate = task_successes / task_episodes if task_episodes > 0 else 0.0
        print(f'  Task {task_id} ({task_desc}): {task_successes}/{task_episodes} = {task_rate:.4f}')

    policy.train()
    return total_successes / total_episodes if total_episodes > 0 else 0.0


def train_bc(train_loader, val_loader, norm_stats, args):
    wandb.init(project='act-libero', config=args)

    device = torch.device(f'cuda:{args["gpu_id"]}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    policy = make_policy(args)
    policy.to(device)
    optimizer = policy.configure_optimizers()

    min_val_loss         = np.inf
    best_ckpt_info       = None
    best_rollout_success = -1.0
    rollout_eval_freq    = args.get('rollout_eval_freq', 0)

    for epoch in tqdm(range(args['num_epochs'])):

        # --train--
        policy.train()
        train_loss = 0.0
        train_l1   = 0.0
        train_kl   = 0.0
        train_grad_norm = 0.0
        for train_data in tqdm(train_loader, desc=f'Epoch {epoch} training'):
            loss_dict = forward_pass(train_data, policy, device)
            loss      = loss_dict['loss']
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float('inf'))                                         
            train_grad_norm += grad_norm.item()  
            optimizer.step()
            train_loss += loss.item()
            train_l1   += loss_dict['l1'].item()
            train_kl   += loss_dict['kl'].item()

        train_loss /= len(train_loader)
        train_l1   /= len(train_loader)
        train_kl   /= len(train_loader)
        train_grad_norm /= len(train_loader) 

        # --val--
        policy.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for val_data in val_loader:
                loss_dict = forward_pass(val_data, policy, device)
                val_loss += loss_dict['loss'].item()
        val_loss /= len(val_loader)

        if val_loss < min_val_loss:
            min_val_loss   = val_loss
            best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))

        print(f'Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, best_val={min_val_loss:.4f}')

        wandb.log({
            'train_loss':  train_loss,
            'val_loss':    val_loss,
            'train_l1':    train_l1,
            'train_kl':    train_kl,
            'lr':          optimizer.param_groups[0]['lr'],
            'lr_backbone': optimizer.param_groups[1]['lr'],
            'train_grad_norm': train_grad_norm,
        }, step=epoch)

        # --rollout eval--
        if rollout_eval_freq > 0 and (epoch + 1) % rollout_eval_freq == 0:
            print(f'\n[Epoch {epoch+1}] Running rollout eval '
                  f'({args.get("num_trials_per_task", 5)} trials/task)...')
            success_rate = rollout_eval(policy, norm_stats, args, device)
            print(f'[Epoch {epoch+1}] Rollout SR: {success_rate:.4f} '
                  f'(best so far: {best_rollout_success:.4f})')
            wandb.log({'rollout/success_rate': success_rate}, step=epoch)

            if success_rate > best_rollout_success:
                best_rollout_success = success_rate
                ckpt_path = os.path.join(args['ckpt_dir'], 'policy_best_rollout.ckpt')
                torch.save({
                    'model_state_dict':    deepcopy(policy.state_dict()),
                    'norm_stats':          norm_stats,
                    'epoch':               epoch + 1,
                    'rollout_success_rate': success_rate,
                }, ckpt_path)
                print(f'[Epoch {epoch+1}] New best rollout! '
                      f'SR={success_rate:.4f} -> policy_best_rollout.ckpt saved')

        if (epoch + 1) % args['save_every'] == 0:
            ckpt_path = os.path.join(args['ckpt_dir'], f'checkpoint_epoch_{epoch+1}.ckpt')
            torch.save({
                'epoch':                epoch,
                'train_loss':           train_loss,
                'val_loss':             val_loss,
                'model_state_dict':     policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'norm_stats':           norm_stats,
            }, ckpt_path)

    torch.save(
        {'model_state_dict': policy.state_dict(), 'norm_stats': norm_stats},
        os.path.join(args['ckpt_dir'], 'policy_last.ckpt')
    )

    if best_ckpt_info is None:
        raise RuntimeError('Training ended before any checkpoint was saved.')

    best_epoch, best_loss, best_state_dict = best_ckpt_info
    torch.save(
        {'model_state_dict': best_state_dict, 'norm_stats': norm_stats},
        os.path.join(args['ckpt_dir'], 'policy_best.ckpt')
    )

    print(f'\n========== Training finished ==========')
    print(f'[Val-loss ckpt]  epoch {best_epoch}, val_loss {best_loss:.6f}')
    if best_rollout_success >= 0:
        print(f'[Rollout ckpt]   best rollout SR = {best_rollout_success:.4f}')
    else:
        print(f'[Rollout ckpt]   rollout eval was disabled (rollout_eval_freq=0)')
    print(f'Eval with val-loss ckpt : --ckpt_name policy_best.ckpt')
    print(f'Eval with rollout ckpt  : --ckpt_name policy_best_rollout.ckpt')
    print(f'=======================================\n')

    wandb.finish()
    return best_ckpt_info


def eval_bc(args, ckpt_name='policy_best.ckpt'):
    wandb.init(project='act-libero', name=f'eval_{ckpt_name}', config=args, job_type='eval')

    device    = torch.device(f'cuda:{args["gpu_id"]}' if torch.cuda.is_available() else 'cpu')
    ckpt_path = os.path.join(args['ckpt_dir'], ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    norm_stats = checkpoint['norm_stats']

    if 'epoch' in checkpoint:
        print(f'Loaded ckpt from epoch {checkpoint["epoch"]}', end='')
        if 'rollout_success_rate' in checkpoint:
            print(f', rollout SR = {checkpoint["rollout_success_rate"]:.4f}')
        else:
            print()

    policy = make_policy(args)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.to(device)
    policy.eval()
    print(f'Loaded: {ckpt_path}')

    video_dir = os.path.join(args['video_dir'], 'video')
    os.makedirs(video_dir, exist_ok=True)

    post_process = lambda a: a * norm_stats['action_std'] + norm_stats['action_mean']

    suite_key      = args['task_suite_name'].replace('_no_noops', '')
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict[suite_key]()
    num_tasks      = task_suite.n_tasks

    num_trials     = args.get('num_trials_per_task_final', 20)
    num_steps_wait = args.get('num_steps_wait', 10)
    num_open_loop  = args.get('num_open_loop_steps', 8)  
    temporal_agg   = args.get('temporal_agg', False)      
    max_steps      = TASK_MAX_STEPS.get(suite_key, 300)
    task_emb_dict = np.load(args['task_emb_path'], allow_pickle=True).item()

    total_episodes, total_successes = 0, 0
    for task_id in range(num_tasks):
        task           = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_desc = get_libero_env(task, 'act', resolution=256)
        task_emb = torch.from_numpy(task_emb_dict[task_desc]).float().unsqueeze(0).to(device)
        print(f'\nTask {task_id}: {task_desc}')

        task_episodes, task_successes = 0, 0
        for trial_idx in tqdm(range(num_trials)):
            initial_state = initial_states[trial_idx % len(initial_states)]
            env.reset()
            obs          = env.set_init_state(initial_state)
            t            = 0
            success      = False
            replay_images = []
            action_queue  = deque()
            num_queries   = args['num_queries']

            if temporal_agg:
                all_time_actions = np.zeros((max_steps, max_steps + num_queries, 7))

            while t < max_steps + num_steps_wait:
                if t < num_steps_wait:
                    obs, _, _, _ = env.step(get_libero_dummy_action('act'))
                    t += 1
                    continue

                replay_images.append(get_libero_image(obs))
                t_rel = t - num_steps_wait

                if temporal_agg:
                    qpos_tensor, image_tensor = get_obs_tensors(
                        obs, norm_stats, args['camera_names'], device)
                    with torch.inference_mode():
                        action_chunk = policy(qpos_tensor, image_tensor, task_emb=task_emb)
                    action_chunk = action_chunk.squeeze(0).cpu().numpy()
                    action_chunk = post_process(action_chunk)
                    fill_len = min(num_queries, max_steps - t_rel)
                    all_time_actions[t_rel, t_rel:t_rel + fill_len] = action_chunk[:fill_len]
                    actions_for_curr = all_time_actions[:t_rel + 1, t_rel]
                    k = args.get('agg_k', 0.01)
                    weights = np.exp(-k * np.arange(len(actions_for_curr))[::-1])
                    weights = weights / weights.sum()
                    action = (actions_for_curr * weights[:, None]).sum(axis=0)
                else:
                    if len(action_queue) == 0:
                        qpos_tensor, image_tensor = get_obs_tensors(
                            obs, norm_stats, args['camera_names'], device)
                        with torch.inference_mode():
                            action_chunk = policy(qpos_tensor, image_tensor, task_emb=task_emb)
                        action_chunk = action_chunk.squeeze(0).cpu().numpy()
                        action_chunk = post_process(action_chunk)
                        for i in range(min(num_open_loop, len(action_chunk))):
                            action_queue.append(action_chunk[i])
                    action = action_queue.popleft()

                obs, _, done, _ = env.step(action.tolist())
                if done:
                    success = True
                    break
                t += 1

            task_episodes  += 1
            total_episodes += 1
            if success:
                task_successes  += 1
                total_successes += 1

            save_rollout_video(replay_images, total_episodes,
                               success=success, task_description=task_desc,
                               save_dir=args['video_dir'])
            print(f'Trial {trial_idx+1}: {"SUCCESS" if success else "FAIL"} | '
                  f'{total_successes}/{total_episodes}')

        task_rate = task_successes / task_episodes if task_episodes > 0 else 0.0
        print(f'Task {task_id} success rate: {task_rate:.4f}')
        wandb.log({f'eval/task_{task_id}_success_rate': task_rate, 'task_id': task_id})

    final_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    print(f'\nOverall success rate: {final_rate:.4f} ({total_successes}/{total_episodes})')
    wandb.log({'eval/final_success_rate': final_rate})
    wandb.finish()
    return final_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true',
                        help='run eval instead of training')
    parser.add_argument('--ckpt_name', type=str, default='policy_best.ckpt',
                        help='checkpoint filename to load for eval')
    parsed = parser.parse_args()

    args = ACT_LIBERO_CONFIG
    torch.manual_seed(args['seed'])
    os.makedirs(args['ckpt_dir'], exist_ok=True)

    if parsed.eval:
        eval_bc(args, ckpt_name=parsed.ckpt_name)
        return

    train_loader, val_loader, norm_stats = load_data(
        data_root=args['data_root'],
        task_suite_name=args['task_suite_name'],
        camera_names=args['camera_names'],
        num_queries=args['num_queries'],
        batch_size_train=args['batch_size'],
        batch_size_eval=args['batch_size'],
        task_emb_path=args['task_emb_path'],
    )
    train_bc(train_loader, val_loader, norm_stats, args)


if __name__ == '__main__':
    main()