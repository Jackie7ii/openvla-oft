ACT_LIBERO_CONFIG = {
    # data
    'data_root':        '/local_data/jl17265/projects/openvla-oft/modified_libero_rlds',
    'task_suite_name':  'libero_spatial_no_noops',
    'camera_names':     ['images', 'wrist_images'],
    'task_emb_path':   '/local_data/jl17265/projects/openvla-oft/experiments/robot/act/task_embeddings/libero_spatial_task_embeddings.npy',


    # model
    'backbone':           'resnet18',
    'position_embedding': 'sine',
    'masks':              False,
    'dilation':           False,
    'hidden_dim':         256,
    'dim_feedforward':    2048,
    'nheads':             8,
    'enc_layers':         4,
    'dec_layers':         6,
    'pre_norm':           False,
    'num_queries':        75,
    'action_dim':         7,
    'qpos_dim':           8,
    'dropout':           0.1,
    'multi_task':          True,  # whether to use a single multi-task model or separate models per task
    'task_emb_dim':       384,   # dimension of task embedding (if multi_task is True)


    # training
    'lr':           1e-5,
    'lr_backbone':  5e-5,
    'weight_decay': 1e-4,
    'kl_weight':    10,
    'batch_size':   8,
    'num_epochs':   5000, 

    # eval
    'temporal_agg':       False,
    'agg_k':              0.1,   # exponential decay factor for temporal aggregation weights
    'num_open_loop_steps':  8,    # number of steps to run open-loop during rollout eval (0 = fully closed-loop)
    'eval_seed' : 0,                 # seed for eval rollout (affects env initialization and action noise)
    'num_trials_per_task_final': 20,          # number of rollout eval trials per task at the end
    
    #eval while training
    'rollout_eval_freq':  100,   # run rollout eval every N epochs (0 = disabled)
    'num_trials_per_task': 5,    # trials per task during in-training rollout eval

    # save
    'ckpt_dir':   '/local_data/jl17265/projects/openvla-oft/mtact_ckpt/libero_spatial_0',
    'video_dir':  '/local_data/jl17265/projects/openvla-oft/mtact_ckpt/libero_spatial_0',
    'save_every': 100,

    # seed
    'seed': 0,

    #gpu
    'gpu_id':7,
}

TASK_MAX_STEPS = {
    'libero_spatial': 220,
    'libero_object':  280,
    'libero_goal':    300,
    'libero_10':      520,
    'libero_90':      400,
}
