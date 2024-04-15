from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.utils as vutils
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from dataset import erase_and_norm, normalize, unnormalize


class Experiment:
    def __init__(
        self,
        model,
        params,
        device='gpu'
    ):
        self.model = model
        self.params = params
        self.device = torch.device(device)

        self.optimizer = Adam(self.model.parameters(), lr=self.params['LR'])
        self.scheduler = ExponentialLR(self.optimizer, gamma=self.params['scheduler_gamma'])
        self.num_steps = 0
        self.log_dir = os.path.join(self.params['save_dir'], self.params['version'])
        
        self.writer = SummaryWriter(self.log_dir)
    
    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)

    def training_step(self, batch):
        batch = normalize(batch)
        batch = batch.to(self.device)

        scores = self.forward(batch)


        train_loss = self.model.loss_function(results, batch)
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()
        self.num_steps += 1

        self.writer.add_scalar('loss', train_loss, global_step=self.num_steps)

        return train_loss

    def sample_images(self, batch_ng, batch_ok):  # batch = next(iter(test_dataloader))  bs=4
        self.model.eval()
        nrow = batch_ng.size(0)
        ng = batch_ng.to(self.device)
        batch_ok = erase_and_norm(batch_ok)
        ok = batch_ok.to(self.device)
        with torch.no_grad():
            repair = unnormalize(self.model.reconstruct(ng))
            recons = unnormalize(self.model.reconstruct(ok))
        contrast = (repair - unnormalize(ng)).abs() ** 0.5
        ng_repair_recons = torch.cat([unnormalize(ng), repair, contrast, recons], dim=0)
        sample_path = os.path.join(
                self.log_dir,
                'Samples',
                f'{self.params["version"]}-Step_{self.num_steps:06d}.png'
        )
        vutils.save_image(
            ng_repair_recons.cpu().data,
            sample_path,
            normalize=True,
            nrow=nrow
        )
        
        self.model.train()

        return sample_path

    def checkpoint(self, epoch):
        torch.save(
            self.model.state_dict(),
            os.path.join(
                self.log_dir,
                'Samples',
                f'{self.params["version"]}-Epoch_{epoch:04d}.pth'
            )
        )

    def log_epoch(self, epoch):
        self.writer.add_scalar('Epoch', epoch, global_step=self.num_steps)
