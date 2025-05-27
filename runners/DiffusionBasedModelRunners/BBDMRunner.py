import os

import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn

from PIL import Image
from Register import Registers
from model.BrownianBridge.BrownianBridgeModel_c2v import BrownianBridgeModel_c2v

from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import weights_init, get_optimizer, get_dataset, make_dir, get_image_grid
from tqdm.autonotebook import tqdm
from torchsummary import summary
import random


@Registers.runners.register_with_name('c2v_BBDMRunner')
class c2v_BBDMRunner(DiffusionBaseRunner):
    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, config):
        if config.model.model_type == "BBDM_c2v":
            bbdmnet = BrownianBridgeModel_c2v(config.model).to(config.training.device[0])
        else:
            raise NotImplementedError
        bbdmnet.apply(weights_init)
        return bbdmnet

    def load_model_from_checkpoint(self):
        states = super().load_model_from_checkpoint()


    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(net)
        print("Total Number of parameter: %.2fM" % (total_num / 1e6))
        print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

    def initialize_optimizer_scheduler(self, net, config):
        optimizer = get_optimizer(config.model.BB.optimizer, net.get_parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               verbose=True,
                                                               threshold_mode='rel',
                                                               **vars(config.model.BB.lr_scheduler))
        return [optimizer], [scheduler]
    

    @torch.no_grad()
    def get_checkpoint_states(self, stage='epoch_end'):
        model_states, optimizer_scheduler_states = super().get_checkpoint_states()
        return model_states, optimizer_scheduler_states

    
    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        (x, x_name), (x_cond, x_cond_name), *_ = batch
        x = x.to(self.config.training.device[0])
        x_cond = x_cond.to(self.config.training.device[0])

        if len(batch) > 3:
            (condition, condition_name) = batch[2]
            condition = condition.to(self.config.training.device[0])
        else:
            condition = None

        loss, additional_info = net(x, x_cond, condition=condition)
        
        if write:
            self.writer.add_scalar(f'loss/{stage}', loss, step)
            if additional_info.__contains__('recloss_noise'):
                self.writer.add_scalar(f'recloss_noise/{stage}', additional_info['recloss_noise'], step)
            if additional_info.__contains__('recloss_xy'):
                self.writer.add_scalar(f'recloss_xy/{stage}', additional_info['recloss_xy'], step)
        return loss
    
    
    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))
        reverse_sample_path = make_dir(os.path.join(sample_path, 'reverse_sample'))
        reverse_one_step_path = make_dir(os.path.join(sample_path, 'reverse_one_step_samples'))

        (x, x_name), (x_cond, x_cond_name), *_ = batch

        batch_size = x.shape[0] if x.shape[0] < 4 else 4

        x = x[0:batch_size].to(self.config.training.device[0])
        x_cond = x_cond[0:batch_size].to(self.config.training.device[0])

        if len(batch) > 3:
            (condition, condition_name) = batch[2]
            condition = condition[0:batch_size].to(self.config.training.device[0])
        else:
            condition = None

        grid_size = 4

        sample = net.sample(x_cond, condition=condition, clip_denoised=self.config.testing.clip_denoised).to('cpu')
        image_grid = get_image_grid(sample[:, :, sample.shape[0]//2, :], grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'skip_sample.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_skip_sample', image_grid, self.global_step, dataformats='HWC')

        image_grid = get_image_grid(x_cond[:, :, x_cond.shape[0]//2, :].to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'condition.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_condition', image_grid, self.global_step, dataformats='HWC')

        image_grid = get_image_grid(x[:, :, x.shape[0]//2, :].to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'ground_truth.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_ground_truth', image_grid, self.global_step, dataformats='HWC')

    
    @torch.no_grad()
    def sample_results_c2v(self, net, test_batch):
        (x, x_name) = test_batch[0]
        (x_cond, x_cond_name) = test_batch[1]
        x = x.to(self.config.training.device[0])
        x_cond = x_cond.to(self.config.training.device[0])

        if len(test_batch) > 2:
            (condition, condition_name) = test_batch[2]
            condition = condition.to(self.config.training.device[0])
        else:
            condition = None

        sample = net.sample(x_cond, condition=condition, clip_denoised=self.config.testing.clip_denoised)
        
        return x, sample

