import numpy as np
import os
from constants import *
from panet import PAN
import argparse, time
from utils.net_utils import adjust_learning_rate
import torch
from torch.autograd import Variable
from dataset.kitti_dataset import KittiDataset
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from path import Path
from torch.utils.data.sampler import Sampler

class RMSE_log(nn.Module):
    def __init__(self):
        super(RMSE_log, self).__init__()
        self.eps = 1e-8
    
    def forward(self, fake, real):
        mask = real>0
        fake += self.eps
        n = len(real[mask])
        loss = torch.sqrt( torch.sum( torch.abs(torch.log(real[mask])-torch.log(fake[mask])) ** 2 ) / n )
        return loss

class iRMSE(nn.Module):
    def __init__(self):
        super(iRMSE, self).__init__()
        self.eps = 1e-8
    
    def forward(self, fake, real):
        mask = real>0
        n = len(real[mask])
        loss = torch.sqrt( torch.sum( torch.abs(real[mask]-fake[mask]) ** 2 ) / n + self.eps )
        return loss
    
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Single image depth estimation')
    parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='kitti', type=str)
    parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=NUM_EPOCHS, type=int)
    parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
    parser.add_argument('--bs', dest='bs',
                      help='batch_size',
                      default=5, type=int)
    parser.add_argument('--num_workers', dest='num_workers',
                      help='num_workers',
                      default=1, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                      help='display interval',
                      default=10, type=int)
    parser.add_argument('--output_dir', dest='output_dir',
                      help='output directory',
                      default='saved_models', type=str)

# config optimization
    parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=1e-3, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
    parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
    parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
    parser.add_argument('--start_at', dest='start_epoch',
                      help='epoch to start with',
                      default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)

# training parameters
    parser.add_argument('--gamma_sup', dest='gamma_sup',
                      help='factor of supervised loss',
                      default=1., type=float)
    parser.add_argument('--gamma_unsup', dest='gamma_unsup',
                      help='factor of unsupervised loss',
                      default=1., type=float)
    parser.add_argument('--gamma_reg', dest='gamma_reg',
                      help='factor of regularization loss',
                      default=10., type=float)

    args = parser.parse_args()
    return args

def get_coords(b, size, use_cuda):
    h, w = size
    i_range = Variable(torch.arange(0, h).view(1, h, 1).expand(b,1,h,w))  # [B, 1, H, W]
    j_range = Variable(torch.arange(0, w).view(1, 1, w).expand(b,1,h,w))  # [B, 1, H, W]
    coords = torch.cat((j_range, i_range), dim=1)
    norm = Variable(torch.Tensor([w,h]).view(1,2,1,1))
    coords = coords * 2. / norm - 1.
    coords = coords.permute(0, 2, 3, 1)
    if use_cuda:
        coords = coords.cuda()
    return coords
        
def resize_tensor(img, coords):
    return nn.functional.grid_sample(img, coords, mode='bilinear', padding_mode='zeros')
    
def berhu_loss(real, fake, coords):
    mask = real>0
    fake = resize_tensor(fake, coords) * mask
    diff = torch.abs(real-fake)
    delta = 0.2 * torch.max(diff).data.cpu().numpy()[0]
    
    part1 = -F.threshold(-diff, -delta, 0.)
    part2 = F.threshold(diff**2 - delta**2, 0., -delta**2.) + delta**2
    part2 = part2 / (2.*delta)
    
    loss = part1 + part2
    loss = torch.sum(loss)
    return loss
    
def l1_loss(real, fake):
    return torch.sum( torch.abs(real-fake) )

def imgrad(img, use_cuda=False):
    img = torch.mean(img, 1, True)
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if use_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if use_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

#     grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))
    
    return grad_y, grad_x

def imgrad_yx(img, use_cuda=False):
    grad_y, grad_x = imgrad(img, use_cuda)
    b,_,_,_ = grad_y.size()
    return torch.cat((grad_y.view(b,-1,1), grad_x.view(b,-1,1)), dim=2)

def reg_scalor(grad_yx):
    return torch.exp(-torch.abs(grad_yx)/255.)
    
    
    
class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

    args = parse_args()

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You might want to run with --cuda")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # dataset
    if args.dataset == 'kitti':
        dataset = KittiDataset(train=True)
        train_size = len(dataset)

        batch_sampler = sampler(train_size, args.bs)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs,
                                sampler=batch_sampler, num_workers=args.num_workers)
    elif args.dataset == 'scannet':
        pass

    # tensor placeholders
    img = torch.FloatTensor(1)
    z = torch.FloatTensor(1)
    
    # network initialization
    print('Initializing model...')
    i2d = PAN()
    print('Done!')

    # cuda
    if args.cuda:
        img = img.cuda()
        z = z.cuda()
        i2d = i2d.cuda()

    # make variables
    img = Variable(img)
    z = Variable(z)

    # hyperparams
    lr = args.lr
    bs = args.bs
    lr_decay_step = args.lr_decay_step
    lr_decay_gamma = args.lr_decay_gamma

    # params
    params = []
    for key, value in dict(i2d.named_parameters()).items():
      if value.requires_grad:
        if 'bias' in key:
          params += [{'params':[value],'lr':lr*(DOUBLE_BIAS + 1), \
                  'weight_decay': 4e-5 and WEIGHT_DECAY or 0}]
        else:
          params += [{'params':[value],'lr':lr, 'weight_decay': 4e-5}]

    # optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

    criterion = RMSE_log()
    eval_metric = iRMSE()
    
    # resume
    if args.resume:
        load_name = os.path.join(output_dir,
          'i2d_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        i2d.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    # constants
    iters_per_epoch = int(train_size / args.bs)
    NUM_PIXELS = TRAIN_IMG_SIZE[0] * TRAIN_IMG_SIZE[1]
    
    gamma_sup = args.gamma_sup / NUM_PIXELS
    gamma_unsup = args.gamma_unsup / NUM_PIXELS
    gamma_reg = args.gamma_reg / NUM_PIXELS
    
    z_coords = get_coords(bs, TRAIN_IMG_SIZE, args.cuda)
    
    for epoch in range(args.max_epochs):
        # setting to train mode
        i2d.train()
        start = time.time()
        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            data = next(data_iter)
            img.data.resize_(data[0].size()).copy_(data[0])
            z.data.resize_(data[1].size()).copy_(data[1])

            optimizer.zero_grad()
            z_fake = i2d(img)
            
            z_fake = resize_tensor(z_fake, z_coords)
            loss = criterion(z_fake, z)
            metric = eval_metric(z_fake, z)

            loss.backward()
            optimizer.step()
            
            # info
            if step % args.disp_interval == 0:
                end = time.time()

                print("[epoch %2d][iter %4d] loss: %.4f iRMSE: %.4f" \
                                % (epoch, step, loss, metric))
             #    

            # save model
        save_name = os.path.join(args.output_dir, 'i2d_{}_{}.pth'.format(args.session, epoch))
        torch.save({
          'session': args.session,
          'epoch': epoch,
        }, save_name)

        print('save model: {}'.format(save_name))
        print('time elapsed: %fs' % (end - start))