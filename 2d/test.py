import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_fairseg import FairSeg_dataset, TestGenerator
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from inference import inference



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='/output/sam/results')
    parser.add_argument('--datadir', type=str,
                        default='/data/home/tiany/Datasets/fair_segmentation_1w_replace_544_white', help='root dir for validation volume data')  
    parser.add_argument('--dataset', type=str,
                        default='FairSeg', help='experiment_name')
    parser.add_argument('--num_classes', type=int,
                        default=3, help='output channel of network')
    parser.add_argument('--list_dir', type=str,
                        default='/data/home/tiany/Projects/project_TransUNet/TransUNet/lists/FairSeg_SLO_replace_544_whole_data', help='list dir')
    parser.add_argument('--attribute', type=str, default='race', help='attribute labels')
    parser.add_argument('--center_crop_size', type=int, default=512, help='center croped image size | 512 for slo, 420 for oct fundus')
    parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
    parser.add_argument('--stop_epoch', type=int, default=160, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='batch_size per gpu')
    parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
    parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

    parser.add_argument('--n_skip', type=int, default=0, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')

    parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--dmoe', type=bool, default=False)
    parser.add_argument('--vmoe', type=bool, default=False)
    parser.add_argument("--screw", default=4, type=int) 
    args = parser.parse_args()


    if not args.deterministic:
        cudnn.benchmark = Truea
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'FairSeg': {
            'Dataset': FairSeg_dataset,
            'data_dir': args.datadir,
            'num_classes': args.num_classes,
        },
    }
    dataset_name = args.dataset
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size) 
    snapshot_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    if (args.list_dir.find('FairSeg_final') >= 0) & (args.screw < 4):
        snapshot_path = snapshot_path + '_screw_' + str(args.screw)

    if args.attribute == 'race' or args.attribute == 'language':
        no_of_attr = 3
    elif args.attribute == 'gender':
        no_of_attr = 2
    elif args.attribute == 'local':
        no_of_attr = 10
    else:
        no_of_attr = 5
    
    print(no_of_attr)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.n_attr = no_of_attr
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes, dmoe=args.dmoe).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.stop_epoch-1))
    
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]
    
    log_folder = f'{snapshot_path}/results/test_log_' + args.exp + '_' + str(args.vit_name)+ '_epoch' + str(args.stop_epoch-1)

    if args.is_savenii:
        args.test_save_dir = '../predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    
    inference(args, net, dataset_config[dataset_name], test_save_path, no_of_attr, log_folder)


