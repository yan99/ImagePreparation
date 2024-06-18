import argparse
import os
import time
import torch
import numpy as np

from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torch import nn
from torch.optim import Adam, RMSprop
import cv2

import DDVM_model
import DDVM_data
import DDVM_utils
import sampling

from DDVM_data import img_w, img_h

def main():
    parser = argparse.ArgumentParser(description='DDVM Denoise')
    # Data
    parser.add_argument('--data-dir', default=None, metavar='DIR',
                        help='path to data')
    parser.add_argument('--T', '--noise_steps', default=200, type=int,
                        help='number of diffusion steps (default: 200)')
    # Train paramters
    parser.add_argument("--cuda_idx", default=0, type=int, help="cuda id")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=1500, type=int, metavar='N',
                        help='number of total epochs to run (default: 1500)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=5, type=int,
                        metavar='N', help='mini-batch size (default: 5)')
    parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print batch frequency (default: 10)')
    parser.add_argument('--save-epoch-freq', '-s', default=50, type=int,
                        metavar='N', help='save epoch frequency (default: 10)')
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

    train, test = DDVM_data.dataset_creation(load_dataset, data_path = None, training_size_portion = 0.8)

    train = train.to_numpy()
    test = test.to_numpy()

    train_data = DDVM_data.Dataset_RGBD(dataframe = train, 
                                            transform = transforms.Compose([DDVM_data.scaleNorm(),
                                                                            DDVM_data.interpolate_mask_x0(),
                                                                            DDVM_data.RandomScale((1.0,1.4)),
                                                                            DDVM_data.RandomCrop(img_h, img_w),
                                                                            DDVM_data.RandomFlip(),
                                                                            DDVM_data.ToTensor(),
                                                                            DDVM_data.Normalize(),]),)
    
    training_loader = DataLoader(train_data,batch_size=args.batch_size, shuffle=True)

#     training_loader = DataLoader(train_data,batch_size=args.batch_size, shuffle=True, 
#                                  num_workers = args.workers, persistent_workers=True, pin_memory=False)

    # for number of images output
    num = 5
    
    num_train = len(train_data)
    # models  
    model = DDVM_model.SimpleUnet()
    # Optimizer
    opt = Adam(params=model.parameters(), lr = args.lr, betas = (args.beta1, args.beta2))
        
    if not os.path.exists("MSMF_clean_depth"):
        os.mkdir("MSMF_clean_depth")
    
    training_loss = []
    # initial weights
    model.apply(DDVM_utils.weight_inits)
    # MSE Loss 
    # criterion = nn.MSELoss(reduction='sum')        
    
    # Forward diffusion
    fd = DDVM_utils.forwardDiffusion(args.T)

    print(f"Training job starts on {device}.........")
    model.to(device)
    model.train()

    global_step = 0

    if args.last_ckpt:
        global_step, args.start_epoch = DDVM_utils.load_ckpt(model, opt, args.last_ckpt, device)

    for epoch in range(int(args.start_epoch), args.epochs):
        local_count = 0
        last_count = 0
        end_time = time.time()
            
        running_loss = 0.0
        for i, data in enumerate(training_loader):
            rgb = data['rgb'].to(device)
            depth = data['depth'].to(device)
            gt = data['gt_mask'].to(device)
            opt.zero_grad()
            
            t = torch.randint(0, args.T, (args.batch_size,), device=device)
            x_t, epsilon = DDVM_utils.forwardDiffusion.forward_diffusion_sample(fd, depth, t, device=device)
            # N C H W
            input = torch.cat((rgb, x_t), 1)
            loss = DDVM_utils.get_loss(model, input, epsilon, gt, t)
            loss.backward()
            opt.step()
            
            running_loss += loss.item()
            local_count += rgb.data.shape[0]
            global_step += 1
            if global_step % args.print_freq == 0 or global_step == 1:
                time_inter = time.time() - end_time
                count_inter = local_count - last_count
                DDVM_utils.print_log(global_step, epoch, local_count, count_inter,
                          num_train, loss, time_inter)
                end_time = time.time()
                print("[Epoch: %d, batch: %5d] Loss: %3f" % (epoch+1, i+1, running_loss/100))
                training_loss.append(running_loss/100)
                running_loss = 0.0
                last_count = local_count

            if(i==0 and epoch % args.save_epoch_freq==0):
                rgb_exp = rgb[0].cpu().detach().numpy()
                rgb_exp = np.expand_dims(rgb_exp, 0)
                rgb_exp = torch.from_numpy(rgb_exp).float()
                out_imgs = sampling.sample_image(num, fd, model, rgb_exp, device = device).numpy()
            
                # out_imgs contains 10 images
                for k in range(num):
                    depth_out = out_imgs[k][0]                    
                    cv2.imwrite(os.path.join(args.ckpt_dir+'/img/'+str(epoch)+'_denoised_'+str(k)+'.png'),depth_out)

                out_rgb = np.zeros((img_h, img_w, 3))
                out_rgb = cv2.normalize(rgb[0].cpu().detach().numpy(),out_rgb,0,255,cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(os.path.join(args.ckpt_dir+'/img/'+str(epoch)+'_RGB.png'), out_rgb)

                out_gt = gt[0][0].cpu().detach().numpy() * depth[0][0].cpu().detach().numpy()
                cv2.imwrite(os.path.join(args.ckpt_dir+'/img/'+str(epoch)+'_gt.png'), out_gt) 
        
        if epoch % args.save_epoch_freq == 0 and epoch != args.start_epoch:
            DDVM_utils.save_ckpt(args.ckpt_dir, model, opt, global_step, epoch,
                    local_count, num_train, training_loss)
    
    DDVM_utils.save_ckpt(args.ckpt_dir, model, opt, global_step, args.epochs, 0, num_train, training_loss)
    with open('loss.csv','a') as loss_out:
        loss_out.write(training_loss)
    print("Training job finish!\n")

if __name__ == '__main__':
    main()