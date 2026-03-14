# ACT-LIBERO 使用说明

## 文件结构

```
experiments/robot/act/
├── config.py             # 所有参数配置
├── runact_libero.py      # 训练 + 评估主入口
├── datasetdeal.py        # 数据集加载与缓存
├── policy.py             # ACT policy 定义
├── detr/
│   ├── detr_vae.py       # CVAE 模型主体
│   ├── backbone.py       # ResNet 视觉骨干网络
│   ├── transformer.py    # Transformer 编解码器
│   └── position_encoding.py  # 位置编码
└── utils/
    └── check_npz.py      # 查看 npz 缓存文件内容
```

---

## 数据加载机制

第一次运行时，程序从 RLDS（TFRecord）格式加载数据，并自动将每条 episode 保存为 `.npz` 文件缓存到：

```
{data_root}/{task_suite_name}_cache/
├── episode_0000.npz
├── episode_0001.npz
└── ...
```

之后每次运行直接从 npz 缓存加载，跳过 TFRecord 解析，速度更快。

如需重新生成缓存，手动删除该目录即可。

---

## 第一步：配置参数

所有参数在 `config.py` 里修改。

| 参数 | 说明 | 默认值 |
|---|---|---|
| `data_root` | RLDS 数据集根目录 | `/local_data/.../modified_libero_rlds` |
| `task_suite_name` | 数据集名称 | `libero_spatial_no_noops` |
| `camera_names` | 使用的相机 | `['images', 'wrist_images']` |
| `num_queries` | action chunk 大小 | `20` |
| `batch_size` | 训练 batch 大小 | `8` |
| `num_epochs` | 训练轮数 | `2000` |
| `lr` | 学习率 | `1e-4` |
| `lr_backbone` | backbone 学习率 | `1e-5` |
| `kl_weight` | KL loss 权重 | `10` |
| `ckpt_dir` | checkpoint 保存目录 | `./ckpt/act_libero_spatial0` |
| `save_every` | 每隔多少 epoch 保存一次 | `5` |
| `gpu_id` | 使用第几块 GPU | `1` |
| `temporal_agg` | 是否启用 Temporal Aggregation | `True` |
| `seed` | 随机种子 | `42` |

---

## 第二步：训练

从项目根目录运行：

```bash
cd /local_data/jl17265/projects/openvla-oft
python -m experiments.robot.act.runact_libero
```

训练过程会打印：

```
Epoch 0: train_loss=x.xxxx, val_loss=x.xxxx, best_val_loss=x.xxxx
```

训练曲线同步到 wandb 项目 `act-libero`。

checkpoint 保存在 `ckpt_dir` 目录下：

- `checkpoint_epoch_5.ckpt`、`checkpoint_epoch_10.ckpt` ...（每隔 `save_every` 保存）
- `policy_last.ckpt`（训练结束时的最后一个）
- `policy_best.ckpt`（val loss 最低时的最优模型）

---

## 第三步：评估

默认加载 `policy_best.ckpt`：

```bash
python -m experiments.robot.act.runact_libero --eval
```

指定其他 checkpoint：

```bash
python -m experiments.robot.act.runact_libero --eval --ckpt_name checkpoint_epoch_500.ckpt
```

评估会对每个 task 跑 `num_trials_per_task` 次，打印每次结果和最终成功率。

---

## Temporal Aggregation

在 `config.py` 里切换：

```python
'temporal_agg': True,   # 每步都 query，对历史预测指数加权平均，动作更平滑
'temporal_agg': False,  # open-loop，每隔 num_open_loop_steps 步 query 一次
```
