ACT_LIBERO_CONFIG = {
    # data
    'data_root':        '/local_data/jl17265/projects/openvla-oft/modified_libero_rlds',
    'task_suite_name':  'libero_spatial_no_noops',
    'camera_names':     ['images', 'wrist_images'],


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
    'num_queries':        100,
    'action_dim':         7,
    'qpos_dim':           8,
    'dropout':           0.1,

    # training
    'lr':           1e-4,
    'lr_backbone':  1e-5,
    'weight_decay': 1e-4,
    'kl_weight':    10,
    'batch_size':   8,
    'num_epochs':   2000,

    # eval
    'temporal_agg': True ,

    # save
    'ckpt_dir':   '/local_data/jl17265/projects/openvla-oft/experiments/robot/act/ckpt/act_libero_spatial0',
    'save_every': 50,

    # seed
    'seed': 42,

    #gpu
    'gpu_id': 1,
}

TASK_MAX_STEPS = {
    'libero_spatial': 220,
    'libero_object':  280,
    'libero_goal':    300,
    'libero_10':      520,
    'libero_90':      400,
}
