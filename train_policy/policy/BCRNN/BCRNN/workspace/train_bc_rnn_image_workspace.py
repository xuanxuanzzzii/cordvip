if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from BCRNN.workspace.base_workspace import BaseWorkspace
from BCRNN.dataset.base_dataset import BaseImageDataset
from BCRNN.common.checkpoint_util import TopKCheckpointManager
from BCRNN.common.json_logger import JsonLogger
from BCRNN.common.pytorch_util import dict_apply, optimizer_to


OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainRobomimicImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        
        wandb.init(
            project = ('bc_rnn' + '_' + self.cfg.task.name + '_' + 
                str(self.cfg.expert_data_num) + '_' + 
                str(self.cfg.training.num_epochs) + '_' + 
                str(self.cfg.horizon) + '_' + 
                str(self.cfg.n_obs_steps) + '_' + 
                str(self.cfg.n_action_steps) + '_' + 
                str(self.cfg.crop_h) + '_' +
                str(self.cfg.crop_w) ), 
            config=OmegaConf.to_container(cfg, resolve=True),  
            dir=str(self.output_dir)  
        )

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: RobomimicImagePolicy = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = create_dataloader(dataset, **cfg.dataloader)
        # train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = create_dataloader(val_dataset, **cfg.val_dataloader)
        # val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)

        # # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)
        env_runner = None

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dataset.postprocess(batch, device)
                        # batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        info = self.model.train_on_batch(batch, epoch=self.epoch)

                        # logging 
                        loss_cpu = info['losses']['action_loss'].item()
                        tepoch.set_postfix(loss=loss_cpu, refresh=False)
                        train_losses.append(loss_cpu)
                        step_log = {
                            'train_loss': loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch
                        }
                        wandb.log({
                            'train_loss': loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch
                        })
                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            json_logger.log(step_log)
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
                self.model.eval()

                # # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(self.model)
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dataset.postprocess(batch, device)
                                # batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                info = self.model.train_on_batch(batch, epoch=self.epoch, validate=True)
                                loss = info['losses']['action_loss']
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                            wandb.log({'val_loss': val_loss, 'epoch': self.epoch})

                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        # batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        # obs_dict = batch['obs']
                        # gt_action = batch['action']

                        batch = train_sampling_batch
                        obs_dict = batch['obs']
                        gt_action = batch['joint_action']

                        T = gt_action.shape[1]

                        pred_actions = list()
                        self.model.reset()
                        for i in range(T):
                            result = self.model.predict_action(
                                dict_apply(obs_dict, lambda x: x[:,[i]])
                            )
                            pred_actions.append(result['action'])
                        pred_actions = torch.cat(pred_actions, dim=1)
                        mse = torch.nn.functional.mse_loss(pred_actions, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        wandb.log({'train_action_mse_error': mse.item()})
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_actions
                        del mse

                # checkpoint
                if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    # save_name = 'bc_rnn' + '_' + (self.cfg.task.name + '_' + 
                    #     str(self.cfg.expert_data_num) + '_' + 
                    #     str(self.cfg.training.num_epochs) + '_' + 
                    #     str(self.cfg.horizon) + '_' + 
                    #     str(self.cfg.n_obs_steps) + '_' + 
                    #     str(self.cfg.n_action_steps) + '_' + 
                    #     str(self.cfg.crop_h) + '_' +
                    #     str(self.cfg.crop_w) )
                    save_name = 'bc_rnn' + '_' + self.cfg.task.name 
                    print ("save_name:",save_name)
                    # self.save_checkpoint(f'checkpoints/{save_name}_{seed}/{self.epoch + 1}.ckpt') # TODO
                    self.save_checkpoint(f'checkpoints/{save_name}/{self.epoch + 1}.ckpt')

                # ========= eval end for this epoch ==========
                self.model.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

class BatchSampler:
    def __init__(self, data_size: int, batch_size: int, shuffle: bool = False, seed: int = 0, drop_last: bool = True):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[:-self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch

def create_dataloader(dataset, *, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool, persistent_workers: bool, seed: int = 0):
    batch_sampler = BatchSampler(len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True)
    def collate(x):
        assert len(x) == 1
        return x[0]
    dataloader = DataLoader(dataset, collate_fn=collate, sampler=batch_sampler, num_workers=num_workers, pin_memory=False, persistent_workers=persistent_workers)
    return dataloader


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainRobomimicImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()