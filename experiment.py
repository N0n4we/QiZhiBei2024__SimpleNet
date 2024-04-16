from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.utils as vutils
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from dataset import normalize, unnormalize
import numpy as np

class Experiment:
    def __init__(
        self,
        model,
        params,
        device='cpu'
    ):
        self.model = model
        self.params = params
        self.device = torch.device(device)
        self.optimizer_proj = Adam(self.model.projection.parameters(), lr=self.params['projLR'])
        self.optimizer_dsc = Adam(self.model.discriminator.parameters(), lr=self.params['dscLR'])
        self.num_steps = 0
        self.log_dir = os.path.join(self.params['save_dir'], self.params['version'])
        
        self.writer = SummaryWriter(str(self.log_dir))
    
    def forward(self, x, mode):
        x = x.to(self.device)
        return self.model(x, mode)

    def training_step(self, batch):
        batch = normalize(batch)

        scores, batchlen = self.forward(batch, mode='train')
        true_scores = scores[:batchlen]
        fake_scores = scores[batchlen:]
        th = self.params['dsc_margin']
        p_true = (true_scores.detach() >= th).sum() / len(true_scores)
        p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
        true_loss = torch.clip(-true_scores + th, min=0)
        fake_loss = torch.clip(fake_scores + th, min=0)
        loss = true_loss.mean() + fake_loss.mean()

        self.optimizer_proj.zero_grad()
        self.optimizer_dsc.zero_grad()
        loss.backward()
        self.optimizer_proj.step()
        self.optimizer_dsc.step()

        self.num_steps += 1

        self.writer.add_scalar('loss', loss, global_step=self.num_steps)
        self.writer.add_scalar('p_true', p_true, global_step=self.num_steps)
        self.writer.add_scalar('p_fake', p_fake, global_step=self.num_steps)

        return loss, p_true, p_fake

    def evaluate_step(self, batch):
        batch = normalize(batch)
        batchsize = batch.shape[0]
        scores, _, patch_shapes = self.forward(batch, mode='eval')
        patch_scores = image_scores = -scores
        patch_scores = patch_scores.cpu().numpy()
        image_scores = image_scores.cpu().numpy()
        image_scores = self.model.patch_maker.unpatch_scores(
            image_scores, batchsize=batchsize
        )
        image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
        image_scores = self.model.patch_maker.score(image_scores)

        patch_scores = self.model.patch_maker.unpatch_scores(
            patch_scores, batchsize=batchsize
        )
        scales = patch_shapes[0]
        patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
        batch = unnormalize(batch)
        features = batch.reshape(batchsize, scales[0], scales[1], -1)
        masks, features = self.model.anomaly_segmentor.convert_to_segmentation(patch_scores, features)

        return image_scores, masks, features

    def sample_images(self, batch):  # batch = next(iter(test_dataloader))  bs=4
        self.model.eval()
        nrow = batch.size(0)
        with torch.no_grad():
            image_scores, masks, features = self.evaluate_step(batch)
        masks = torch.from_numpy(np.stack(masks, axis=0))
        features = torch.from_numpy(np.stack(features, axis=0))
        ng_masks = torch.cat([features, masks], dim=0)
        sample_path = os.path.join(
                str(self.log_dir),
                'Samples',
                f'{self.params["version"]}-Step_{self.num_steps:06d}.png'
        )
        vutils.save_image(
            ng_masks.cpu().data,
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
                str(self.log_dir),
                'Samples',
                f'{self.params["version"]}-Epoch_{epoch:04d}.pth'
            )
        )

    def log_epoch(self, epoch):
        self.writer.add_scalar('Epoch', epoch, global_step=self.num_steps)
