import datetime
import logging
import math
import time
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (
    AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, 
    get_time_str, init_tb_logger, init_wandb_logger, make_exp_dirs, 
    mkdir_and_rename, scandir
)
from basicsr.utils.options import copy_opt_file, dict2str
from polarfree.utils.options import parse_options


def init_tb_loggers(opt):
    """Initialize tensorboard and wandb loggers."""
    # Initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, 'should turn on tensorboard when using wandb'
        init_wandb_logger(opt)
    
    # Initialize tensorboard logger
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    
    return tb_logger


def create_train_val_dataloader(opt, logger):
    """Create train and validation dataloaders."""
    train_loader, val_loaders = None, []
    
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed']
            )
            
            # Calculate training statistics
            # print('样本数',len(train_set))
            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / 
                (dataset_opt['batch_size_per_gpu'] * opt['world_size'])
            )
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / num_iter_per_epoch)
            
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
                        
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, 
                dataset_opt, 
                num_gpu=opt['num_gpu'], 
                dist=opt['dist'], 
                sampler=None, 
                seed=opt['manual_seed']
            )
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')
    
    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    """Load resume state for training continuation."""
    resume_state_path = None
    
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']
    
    if resume_state_path is None:
        return None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            resume_state_path, 
            map_location=lambda storage, loc: storage.cuda(device_id)
        )
        check_resume(opt, resume_state['iter'])
        return resume_state


def setup_training_environment(opt, args):
    """Setup training environment including directories and loggers."""
    resume_state = load_resume_state(opt)
    
    # Create experiment directories if not resuming
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))
    
    # Copy config file
    copy_opt_file(args.opt, opt['path']['experiments_root'])
    
    # Setup logger
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    
    # Initialize tensorboard and wandb loggers
    tb_logger = init_tb_loggers(opt)
    
    return logger, tb_logger, resume_state


def setup_data_prefetcher(train_loader, opt, logger):
    """Setup data prefetcher for efficient data loading."""
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")
    
    return prefetcher


def update_training_settings(current_iter, groups, mini_gt_sizes, mini_batch_sizes, logger_j, logger):
    """Update training settings based on current iteration (progressive learning)."""
    # Find appropriate batch size and patch size based on current iteration
    j = ((current_iter > groups) != True).nonzero()[0]
    if len(j) == 0:
        bs_j = len(groups) - 1
    else:
        bs_j = j[0]
    
    mini_gt_size = mini_gt_sizes[bs_j]
    mini_batch_size = mini_batch_sizes[bs_j]
    
    # Log changes in batch size and patch size
    if logger_j[bs_j]:
        logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(
            mini_gt_size, mini_batch_size * torch.cuda.device_count()))
        logger_j[bs_j] = False
    
    return bs_j


def train_pipeline(root_path):
    """Main training pipeline."""
    # Parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path
    torch.backends.cudnn.benchmark = True
    
    # Setup training environment
    logger, tb_logger, resume_state = setup_training_environment(opt, args)
    
    # Create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result
    
    # Create model
    model = build_model(opt)
    
    # Resume training if needed
    start_epoch = 0
    current_iter = 0
    if resume_state:
        model.resume_training(resume_state)  # Handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    
    # Create message logger and data prefetcher
    msg_logger = MessageLogger(opt, current_iter, tb_logger)
    prefetcher = setup_data_prefetcher(train_loader, opt, logger)
    
    # Setup timers
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()
    
    # Progressive training settings
    iters = opt['datasets']['train'].get('iters', [0])
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes', [opt['datasets']['train'].get('gt_size', 256)])
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes', [opt['datasets']['train'].get('batch_size_per_gpu', 4)])
    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])
    logger_j = [True] * len(groups)
    
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    
    # Training loop
    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()
        
        while train_data is not None:
            data_timer.record()
            current_iter += 1
            
            if current_iter > total_iters:
                break
            
            # Update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            
            # # Update training settings (progressive learning)
            # update_training_settings(current_iter, groups, mini_gt_sizes, mini_batch_sizes, logger_j, logger)

            # Model forward and optimization
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()
            
            if current_iter == 1:
                # Reset start time in msg_logger for more accurate eta_time
                msg_logger.reset_start_time()
            
            # Log training info
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
            
            # Save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)
            
            # Validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
            
            # Fetch next batch
            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
    
    # End of training
    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    # print(root_path)
    train_pipeline(root_path)
