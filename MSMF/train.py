import argparse
import os
import time
import torch
import numpy as np
import pickle
import pandas

from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torch import nn
from torch.optim import Adam, RMSprop
import cv2

import MSMF_model
import MSMF_data
import utils

img_w = 128 # 2880
img_h = 128 # 1988


def main():
    parser = argparse.ArgumentParser(description='RedNet Indoor Sementic Segmentation')
    # Data
    parser.add_argument('--data-dir', default=None, metavar='DIR',
                        help='path to SUNRGB-D')
    # Train paramters
    parser.add_argument("--cuda_idx", default=0, type=int, help="cuda id")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=1500, type=int, metavar='N',
                        help='number of total epochs to run (default: 1500)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=5, type=int,
                        metavar='N', help='mini-batch size (default: 10)')
    parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--print-freq', '-p', default=200, type=int,
                        metavar='N', help='print batch frequency (default: 50)')
    parser.add_argument('--save-epoch-freq', '-s', default=5, type=int,
                        metavar='N', help='save epoch frequency (default: 5)')
    parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--ckpt-dir', default='checkpts/', metavar='DIR',
                        help='path to save checkpoints')
    # optimizer
    parser.add_argument("--beta1", default=0.5, type=float, help="beta 1 of ADAM")
    parser.add_argument("--beta2", default=0.999, type=float, help="beta 2 of ADAM")

    args = parser.parse_args()
    if args.last_ckpt:
        load_dataset = True
    else:
        load_dataset = False
    
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)

    device = torch.device(f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")
    # f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu"

    train, test = MSMF_data.dataset_creation(load_dataset, data_path = None, training_size_portion = 0.8)

    train = train.to_numpy()
    test = test.to_numpy()

    train_data = MSMF_data.middleburyDataset(dataframe = train, 
                                            transform = transforms.Compose([MSMF_data.scaleNorm(),
                                                                            MSMF_data.RandomScale((1.0,1.4)),
                                                                            MSMF_data.RandomCrop(img_h, img_w),
                                                                            MSMF_data.RandomFlip(),
                                                                            MSMF_data.ToTensor(),
                                                                            MSMF_data.Normalize()]),)
    
    training_loader = DataLoader(train_data,batch_size=args.batch_size, shuffle=True)

#     training_loader = DataLoader(train_data,batch_size=args.batch_size, shuffle=True, 
#                                  num_workers = args.workers, persistent_workers=True, pin_memory=False)

    num_train = len(train_data)
    # models
    model = MSMF_model.MSMF_CNN(pretrain = True)
    # Optimizer
    opt = Adam(params=model.parameters(), lr = args.lr, betas = (args.beta1, args.beta2))
        
    if not os.path.exists("MSMF_clean_depth"):
        os.mkdir("MSMF_clean_depth")
    
    training_loss = []
    # initial weights
    model.apply(utils.weight_inits)
    # MSE Loss
    criterion = nn.MSELoss(reduction='sum')        
    
    print(f"Training job starts on {device}.........")
    model.to(device)
    model.train()

    global_step = 0

    if args.last_ckpt:
        global_step, args.start_epoch = utils.load_ckpt(model, opt, args.last_ckpt, device)

    for epoch in range(int(args.start_epoch), args.epochs):
        local_count = 0
        last_count = 0
        end_time = time.time()
            
        running_loss = 0.0
        for i, data in enumerate(training_loader):
            rgb = data['rgb'].to(device)
            depth = data['depth'].to(device)
            gt = data['gt'].to(device)
            opt.zero_grad()
            outputs = model(rgb,depth, pretrain=True)
            loss = criterion(outputs, gt)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            local_count += rgb.data.shape[0]
            global_step += 1
            if global_step % args.print_freq == 0 or global_step == 1:
                time_inter = time.time() - end_time
                count_inter = local_count - last_count
                utils.print_log(global_step, epoch, local_count, count_inter,
                          num_train, loss, time_inter)
                end_time = time.time()
                # print("[Epoch: %d, batch: %5d] Loss: %3f" % (epoch+1, i+1, running_loss/100))
                training_loss.append(running_loss/100)
                running_loss = 0.0
                last_count = local_count

            if(i==0 and epoch % args.save_epoch_freq==0):
                depth_out = depth[0].cpu().detach().numpy()
                outputs_out = outputs[0].cpu().detach().numpy()
                gt_out = gt[0].cpu().detach().numpy()
                
                depth_out = depth_out.transpose((1,2,0))
                outputs_out = outputs_out.transpose((1,2,0))
                gt_out = gt_out.transpose((1,2,0))
                
                depth_out = cv2.normalize(depth_out,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
                outputs_out = cv2.normalize(outputs_out,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
                gt_out = cv2.normalize(gt_out,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite('checkpts/img/depth_10_{}.png'.format(epoch),depth_out)
                cv2.imwrite('checkpts/img/output_10_{}.png'.format(epoch),outputs_out)
                cv2.imwrite('checkpts/img/gt_10_{}.png'.format(epoch),gt_out)
            
        if epoch % args.save_epoch_freq == 0 and epoch != args.start_epoch:
            utils.save_ckpt(args.ckpt_dir, model, opt, global_step, epoch,
                    local_count, num_train, training_loss)
    
    utils.save_ckpt(args.ckpt_dir, model, opt, global_step, args.epochs, 0, num_train, training_loss)
    with open('loss.pickle',"wb") as loss_out:
        pickle.dump(training_loss, loss_out)
    print("Training job finish!\n")

if __name__ == '__main__':
    main()