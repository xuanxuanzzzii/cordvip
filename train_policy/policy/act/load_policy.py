from policy import ACTPolicy, CNNMLPPolicy
import os
import torch

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def get_policy(ckpt_dir="../policy/act/checkpoints/assembly"):
    state_dim = 22
    lr_backbone = 1e-5
    backbone = 'resnet18'
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    ckpt_dir = ckpt_dir
    ckpt_path = os.path.join(ckpt_dir, 'policy_epoch_5000_seed_0.ckpt') #checkpoint文件路径
    policy_class = 'ACT'
    policy_config = {'lr': 10e-5,
                    'num_queries': 30,
                    'kl_weight': 100,
                    'hidden_dim': 512,
                    'dim_feedforward': 3200,
                    'lr_backbone': lr_backbone,
                    'backbone': backbone,
                    'enc_layers': enc_layers,
                    'dec_layers': dec_layers,
                    'nheads': nheads,
                    'camera_names': ['cam0'],
                    }
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    # device = torch.device(device)
    device = torch.device('cuda:0')
    policy.to(device)
    policy.eval()
    return policy