import torch
from simplenet import SimpleNet
from experiment import Experiment
import time
import yaml
import random
import numpy as np
from tqdm import tqdm
from dataset import get_DataLoader
from pathlib import Path
import os
from PIL import Image

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# seed everything
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(config['train_params']['seeds'])

Path(str(os.path.join(config["exp_params"]["save_dir"], config["exp_params"]["version"], 'Samples'))).mkdir(exist_ok=True, parents=True)

use_wandb = config['train_params']['use_wandb']
if use_wandb is True:
    import wandb
    wandb.login(key='fb044de2f69c1a28a75f99d8dcd43f7104cfd136')
    wandb.init(project='QiZhiBei', name=config["exp_params"]['version'])
    
    def wandb_log(loss, step, p_true=None, p_fake=None, epoch=None, mode='train'):
        if mode == 'train':
            wandb.log({'loss': loss}, step = step)
            if epoch is not None:
                wandb.log({'Epoch': epoch}, step = step)
            if p_true is not None:
                wandb.log({'p_true': p_true}, step = step)
            if p_fake is not None:
                wandb.log({'p_fake': p_fake}, step = step)
        elif mode == 'valid':
            wandb.log({'valid_loss': loss}, step = step)
            if epoch is not None:
                wandb.log({'Epoch': epoch}, step = step)


if __name__ == '__main__':
    model = SimpleNet(**config['model_params'])
    experiment = Experiment(model, config['exp_params'])
    train_dataloader, test_dataloader = get_DataLoader(**config['data_params'])
    train_config = config['train_params']
    start_time = time.time()
    print(torch.cuda.get_device_name(0))
    # train
    for epoch in range(1, train_config['max_epochs']):
        print(f"========Epoch {epoch}========")
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for idx, batch in progress_bar:
            train_loss, p_true, p_fake = experiment.training_step(batch)
            
            progress_bar.set_description(f"Processing batch {idx}")
            progress_bar.set_postfix(loss = f'{train_loss.item():.4f}')

            if use_wandb is True:
                wandb_log(train_loss, p_true=p_true, p_fake=p_fake, step = experiment.num_steps, epoch = epoch)
            experiment.log_epoch(epoch)
        # sample
        batch_ng = next(iter(test_dataloader))
        pic_path = experiment.sample_images(batch_ng)
        if use_wandb is True:
            img = Image.open(pic_path)
            wandb.log({'ng_masks': [wandb.Image(img, caption=f'Epoch_{epoch}-step_{experiment.num_steps}')]}, step=experiment.num_steps)
        if epoch % 5 == 0:
            experiment.checkpoint(epoch)
        
    print("--------Training Finished--------")
    end_time = time.time()
    hours = (end_time - start_time) / 3600.0
    print(f"Elapsed Time: {hours:.2f}h")
    if use_wandb is True:
        wandb.finish()
