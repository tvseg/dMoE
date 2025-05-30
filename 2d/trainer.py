import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
import matplotlib.pyplot as plt


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

def trainer_fairseg(args, model, snapshot_path):
    from datasets.dataset_fairseg import FairSeg_dataset, RandomGenerator
    print(snapshot_path)
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    stop_epoch = args.stop_epoch
    # max_iterations = args.max_iterations
    db_train = FairSeg_dataset(base_dir=args.root_path, attr_label=args.attribute, split="train", args=args, 
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size], \
                                                    center_crop_size=args.center_crop_size, use_normalize=True)]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    os.makedirs(snapshot_path + '/vis', exist_ok=True)
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            if args.dmoe:
                dmoe = sampled_batch['attr_label']
                dmoe = dmoe.cuda()
                if args.vmoe:
                    dmoe = torch.zeros_like(dmoe)
            else:
                dmoe = None
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch, dmoe)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            if iter_num % 200 == 0:
                logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))                

        save_interval = int(max_epoch/10)
        if (epoch_num + 1) % save_interval == 0: #epoch_num > int(max_epoch / 2) and
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

            image = image_batch[0, 0:1, :, :]
            image = (image - image.min()) / (image.max() - image.min())
            image_path = os.path.join(snapshot_path, f"vis/train_Image_iter_{epoch_num}.png")
            # print(image.shape)
            plt.imsave(image_path, image.cpu().squeeze().numpy())
            # writer.add_image('train/Image', image, iter_num)
            labs = label_batch[0, ...].unsqueeze(0) * 50
            # print(labs.shape)
            ground_truth_path = os.path.join(snapshot_path, f"vis/train_GroundTruth_iter_{epoch_num}.png")
            plt.imsave(ground_truth_path, labs.cpu().squeeze().numpy())
            # writer.add_image('train/GroundTruth', labs, iter_num)
            output_masks = outputs[0, ...] * 50
            output_masks = torch.argmax(torch.softmax(output_masks, dim=0), dim=0, keepdim=True)
            # print(output_masks.shape)
            prediction_path = os.path.join(snapshot_path, f"vis/train_Prediction_iter_{epoch_num}.png")
            plt.imsave(prediction_path, output_masks.detach().cpu().squeeze().numpy())
            # writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)

        if (epoch_num >= max_epoch - 1): # | (epoch_num >= stop_epoch - 1)
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"