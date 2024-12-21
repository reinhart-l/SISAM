import torch

# 加载两个 .pth 文件
checkpoint_1 = torch.load('/home3/yuchu/MedSAM/work_dir/models/MedSAM_SIS_25epo/epoch12_sam.pth')
checkpoint_2 = torch.load('/home3/yuchu/MedSAM/work_dir/models_endovis_2018/MedSAM/epoch15_sam.pth')

# 获取 state_dict
state_dict_1 = checkpoint_1['model']  # 假设模型保存在 'model' 键下
state_dict_2 = checkpoint_2['model']

# 去掉 state_dict_1 中所有键的 'module.' 前缀
new_state_dict_1 = {key.replace('module.', ''): value for key, value in state_dict_1.items()}

# 比较两个 state_dict 的键
keys_1 = set(new_state_dict_1.keys())
keys_2 = set(state_dict_2.keys())

# 找到各自缺失的键
missing_in_1 = keys_2 - keys_1
missing_in_2 = keys_1 - keys_2

# 打印结果
print("Missing in model 1 after prefix removal:", missing_in_1)
print("Missing in model 2:", missing_in_2)

# 打印 optimizer 的信息（假设 optimizer 也保存在 'optimizer' 键下）
print("Optimizer in checkpoint 1:", checkpoint_1.get('optimizer', 'No optimizer found'))
print("Optimizer in checkpoint 2:", checkpoint_2.get('optimizer', 'No optimizer found'))
