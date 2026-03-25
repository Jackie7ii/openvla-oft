import sys                                                                                                                                                                                                                           
import torch                                                                                                                   
from experiments.robot.act.datasetdeal import load_data                                                                        
from experiments.robot.act.policy import ACTPolicy                                                                             
from experiments.robot.act.config import ACT_LIBERO_CONFIG
                                                                                                                                 
args = ACT_LIBERO_CONFIG
device = torch.device('cpu')  # 用 cpu 快速验证，不需要 gpu                                                                    
                                                                                                                                 
# 1. 加载数据，检查 task_emb 是否出现                                                                                          
train_loader, val_loader, norm_stats = load_data(                                                                              
    data_root=args['data_root'],                                                                                               
    task_suite_name=args['task_suite_name'],
    camera_names=args['camera_names'],                                                                                         
    num_queries=args['num_queries'],                                                                                           
    batch_size_train=2,
    batch_size_eval=2,  
    task_emb_path=args['task_emb_path'],                                                                                                       
  )               
                                                                                                                                 
batch = next(iter(train_loader))
qpos, image, action, is_pad, task_emb = batch
print('qpos:     ', qpos.shape)       # (2, 8)                                                                                 
print('image:    ', image.shape)      # (2, 2, 3, 256, 256)                                                                    
print('action:   ', action.shape)     # (2, 75, 7)                                                                             
print('task_emb: ', task_emb.shape)   # (2, 384)  ← 关键                                                                       
                                                                                                                                 
# 2. 构建模型，做一次前向                                                                                                      
policy = ACTPolicy(args)                                                                                                       
policy.to(device)                                                                                                              
                  
loss_dict = policy(qpos, image, action, is_pad, task_emb)                                                                      
print('loss:     ', loss_dict['loss'].item())
print('l1:       ', loss_dict['l1'].item())                                                                                    
print('kl:       ', loss_dict['kl'].item())
                                                                                                                                 
# 3. 做一次反向，验证梯度                                                                                                      
loss_dict['loss'].backward()
print('backward OK')