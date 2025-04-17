if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import pdb
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
import sys
sys.path.insert(0, '../')
sys.path.append('CordViP/env_runner')
sys.path.append('CordViP/CordViP/policy')
sys.path.append('CordViP')
sys.path.append('CordViP/CordViP')

from hydra.core.hydra_config import HydraConfig
from CordViP.policy.CordViP_train import CordViP
from CordViP.dataset.base_dataset import BaseDataset
from CordViP.env_runner.base_runner import BaseRunner
from CordViP.common.checkpoint_util import TopKCheckpointManager
from CordViP.common.pytorch_util import dict_apply, optimizer_to
from CordViP.model.diffusion.ema_model import EMAModel
from CordViP.model.common.lr_scheduler import get_scheduler
import pdb

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainCordViPWorkspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: CordViP = hydra.utils.instantiate(cfg.policy)

        self.ema_model: CordViP = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)


        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def checkpoint_load(self, load_path):
        if not os.path.exists(load_path):
            print(f"Checkpoint file {load_path} does not exist!")
            return
        
        checkpoint = torch.load(load_path, weights_only=True)
        self.model.obs_encoder.load_state_dict(checkpoint['obs_encoder_state_dict'])
        print(f"Model checkpoint loaded from {load_path}")

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_ROLLOUT = False
        RUN_VALIDATION = True # reduce time cost
        
        
        save_name = 'CordViP_Pretrain'+'_'+cfg.policy.pointnet_type+'_'+'all_tasks'+'_'+str(cfg.n_obs_steps)
       
        load_path = f'./CordViP/checkpoints/{cfg.load_checkpoint_name}/{cfg.load_checkpoint_number}.ckpt'
        self.checkpoint_load(load_path)

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
        
        env_runner = None
        
        cfg.logging.name = str(cfg.task.name)

        # configure logging
        wandb.init(
            project='CordViP_load_pretrain_all_tasks'+'_'+cfg.policy.pointnet_type+'_'+str(cfg.training.freeze_encoder)+'_'+cfg.task.name+'_'+str(cfg.horizon)+'_'+str(cfg.n_obs_steps)+'_'+str(cfg.n_action_steps), 
            # entity='xuanxuanziattju',  
            config=OmegaConf.to_container(cfg, resolve=True),  
            dir=str(self.output_dir)  
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None
        checkpoint_num = 1
        
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
            if cfg.training.freeze_encoder:
                print("freeze")
                self.model.obs_encoder.eval()
                self.model.obs_encoder.requires_grad_(False)
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                
                    # compute loss
                    t1_1 = time.time()
                    raw_loss, loss_dict = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                    
                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)
                    t1_4 = time.time()
                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }
                    wandb.log({
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    })
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()
                    
                    if verbose:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss
            wandb.log({'train_loss_average': train_loss})
            # ========= eval for this epoch ==========
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss, loss_dict = self.model.compute_loss(batch)
                            val_losses.append(loss)
                            print(f'epoch {self.epoch}, eval loss: ', float(loss.cpu()))
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss
                        wandb.log({'val_loss': val_loss, 'epoch': self.epoch})
                
            # checkpoint
            if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                
                if not cfg.policy.use_pc_color:
                    if not os.path.exists(f"checkpoints/{'CordViP_load_pretrain_all_tasks'+'_'+cfg.policy.pointnet_type+'_'+str(cfg.training.freeze_encoder)+'_'+self.cfg.task.name+'_'+str(self.cfg.horizon)+'_'+str(self.cfg.n_obs_steps)+'_'+str(self.cfg.n_action_steps)}"):
                        os.makedirs(f"checkpoints/{'CordViP_load_pretrain_all_tasks'+'_'+cfg.policy.pointnet_type+'_'+str(cfg.training.freeze_encoder)+'_'+self.cfg.task.name+'_'+str(self.cfg.horizon)+'_'+str(self.cfg.n_obs_steps)+'_'+str(self.cfg.n_action_steps)}")
                    save_path = f"checkpoints/{'CordViP_load_pretrain_all_tasks'+'_'+cfg.policy.pointnet_type+'_'+str(cfg.training.freeze_encoder)+'_'+self.cfg.task.name+'_'+str(self.cfg.horizon)+'_'+str(self.cfg.n_obs_steps)+'_'+str(self.cfg.n_action_steps)}/{self.epoch + 1}.ckpt"
                else:
                    if not os.path.exists(f"checkpoints/{'CordViP_load_pretrain_all_tasks'+'_'+cfg.policy.pointnet_type+'_'+str(cfg.training.freeze_encoder)+'_'+self.cfg.task.name+'_'+str(self.cfg.horizon)+'_'+str(self.cfg.n_obs_steps)+'_'+str(self.cfg.n_action_steps)}_w_rgb"):
                        os.makedirs(f"checkpoints/{'CordViP_load_pretrain_all_tasks'+'_'+cfg.policy.pointnet_type+'_'+str(cfg.training.freeze_encoder)+'_'+self.cfg.task.name+'_'+str(self.cfg.horizon)+'_'+str(self.cfg.n_obs_steps)+'_'+str(self.cfg.n_action_steps)}_w_rgb")
                    save_path = f"checkpoints/{'CordViP_load_pretrain_all_tasks'+'_'+cfg.policy.pointnet_type+'_'+str(cfg.training.freeze_encoder)+'_'+self.cfg.task.name+'_'+str(self.cfg.horizon)+'_'+str(self.cfg.n_obs_steps)+'_'+str(self.cfg.n_action_steps)}_w_rgb/{self.epoch + 1}.ckpt"

                self.save_checkpoint(save_path)
                

            # ========= eval end for this epoch ==========
            policy.train()

            # end of epoch
            # log of last step is combined with validation and rollout
            self.global_step += 1
            self.epoch += 1
            del step_log

    def get_policy_and_runner(self, cfg, checkpoint_num=3000):
        # load the latest checkpoint
        
        cfg = copy.deepcopy(self.cfg)
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        
        if not cfg.policy.use_pc_color:
            ckpt_file = pathlib.Path(f'./policy/CordViP/CordViP/checkpoints/{self.cfg.task.name}/{checkpoint_num}.ckpt')
        else:
            ckpt_file = pathlib.Path(f'./policy/CordViP/CordViP/checkpoints/{self.cfg.task.name}_w_rgb/{checkpoint_num}.ckpt')
        print('ckpt file exist:', ckpt_file.is_file())
        
        if ckpt_file.is_file():
            cprint(f"Resuming from checkpoint {ckpt_file}", 'magenta')
            self.load_checkpoint(path=ckpt_file)
        
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()
        return policy, env_runner

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        print('saved in ', path)
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)
            
        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")
            
            

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'CordViP', 'config'))
)
def main(cfg):
    workspace = TrainCordViPWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
