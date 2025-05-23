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
from utils import test_single_volume, test_single_image, equity_scaled_perf, equity_scaled_std_perf
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torchvision import transforms
from fairlearn.metrics import *

class_to_name = {1: 'cup', 2: 'disc'}

def inference(args, model, db_config, test_save_path=None, no_of_attr=3, log_folder="./results.txt"):
    db_test = db_config['Dataset'](base_dir=args.datadir, args=args, split='test', attr_label=args.attribute, \
                                    transform=transforms.Compose([TestGenerator(output_size=[args.img_size, args.img_size], center_crop_size=args.center_crop_size, use_normalize=True)]))
    multimask_output=None
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    os.makedirs(log_folder, exist_ok=True)
    print(log_folder + "/results.txt")
    logging.basicConfig(filename=log_folder + "/results.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    try:
        logging.info("Testing console logging.")
    except Exception as e:
        logging.error(f"Error logging args: {e}")
    logging.info(str(args))
    logging.info("{} test iterations per epoch".format(len(testloader)))
    
    model.eval()
    metric_list = 0.0
    nsd_list = 0.0
    asd_list = 0.0
    hausdorff_list = 0.0

    cdr_overall = 0.0
    auc_overall = 0.0

    cdr_by_attr = [ [] for _ in range(no_of_attr) ]
    
    auc_by_attr = [ [] for _ in range(no_of_attr) ]

    dice_by_attr = [ [] for _ in range(no_of_attr) ]
    hd_by_attr = [ [] for _ in range(no_of_attr) ]
    jc_by_attr = [ [] for _ in range(no_of_attr) ]

    NSD_by_attr = [ [] for _ in range(no_of_attr) ]
    ASD_by_attr = [ [] for _ in range(no_of_attr) ]
    Hausdorff_by_attr = [ [] for _ in range(no_of_attr) ]

    dice_by_attr_cup = [ [] for _ in range(no_of_attr) ]
    hd_by_attr_cup = [ [] for _ in range(no_of_attr) ]
    jc_by_attr_cup = [ [] for _ in range(no_of_attr) ]
    NSD_by_attr_cup = [ [] for _ in range(no_of_attr) ]
    ASD_by_attr_cup = [ [] for _ in range(no_of_attr) ]
    Hausdorff_by_attr_cup = [ [] for _ in range(no_of_attr) ]
    
    
    dice_by_attr_rim = [ [] for _ in range(no_of_attr) ]
    hd_by_attr_rim = [ [] for _ in range(no_of_attr) ]
    jc_by_attr_rim = [ [] for _ in range(no_of_attr) ]

    NSD_by_attr_rim = [ [] for _ in range(no_of_attr) ]
    ASD_by_attr_rim = [ [] for _ in range(no_of_attr) ]
    Hausdorff_by_attr_rim = [ [] for _ in range(no_of_attr) ]
    
    all_preds_rim = []
    all_gts_rim = []
    all_attrs_rim = []
     
    all_preds_cup = []
    all_gts_cup = []
    all_attrs_cup = []

    with open(log_folder + '/result_rim.csv', 'w') as file:
        file.write("\nno,ID,Dice,IoU,Attr")
    with open(log_folder + '/result_cup.csv', 'w') as file:
        file.write("\nno,ID,Dice,IoU,Attr")

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name, attr_label = sampled_batch['image'], sampled_batch['label'], \
            sampled_batch['pid'], sampled_batch['attr_label']
        # print(f'{case_name} - {attr_label}')
        metric_i, cdr_dist, auc_score, nsd_metric, asd_metric, hausdorff_metric, \
            preds_array, gts_array, attrs_array  = test_single_image(image, label, model, classes=args.num_classes,                                 multimask_output=multimask_output,
                                      patch_size=[args.img_size, args.img_size], input_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, \
                                        attr_label=attr_label, idx=i_batch, dmoe_flag=args.dmoe, vmoe=args.vmoe)

        # print(metric_i)
        
        
        all_preds_rim.append(preds_array[0]) 
        all_gts_rim.append(gts_array[0]) 
        all_attrs_rim.append(attrs_array[0])
        
        all_preds_cup.append(preds_array[1]) 
        all_gts_cup.append(gts_array[1]) 
        all_attrs_cup.append(attrs_array[1])

        metric_list += np.array(metric_i)
        nsd_list += np.array(nsd_metric)
        asd_list += np.array(asd_metric)

        auc_overall += auc_score
        cdr_overall += cdr_dist

        attr_label = attr_label.detach().cpu().numpy().item()
        with open(log_folder + '/result_rim.csv', 'a') as file:
            file.write("\n%d,%s,%.3f,%.3f,%d,"%(i_batch, sampled_batch['pid'], metric_i[0][0], metric_i[0][2], attr_label))
        with open(log_folder + '/result_cup.csv', 'a') as file:
            file.write("\n%d,%s,%.3f,%.3f,%d,"%(i_batch, sampled_batch['pid'], metric_i[1][0], metric_i[1][2], attr_label))
        
        if attr_label != -1:
            dice_by_attr[attr_label].append(np.mean(metric_i, axis=0)[0])
            hd_by_attr[attr_label].append(np.mean(metric_i, axis=0)[1])
            jc_by_attr[attr_label].append(np.mean(metric_i, axis=0)[2])
            cdr_by_attr[attr_label].append(cdr_dist)
            auc_by_attr[attr_label].append(auc_score)

            NSD_by_attr[attr_label].append(np.mean(nsd_metric))
            ASD_by_attr[attr_label].append(np.mean(asd_metric))

            # compute for rim 
            dice_by_attr_rim[attr_label].append(metric_i[0][0])
            hd_by_attr_rim[attr_label].append(metric_i[0][1])
            jc_by_attr_rim[attr_label].append(metric_i[0][2])

            NSD_by_attr_rim[attr_label].append(nsd_metric[0])
            ASD_by_attr_rim[attr_label].append(asd_metric[0])

            # compute for cup 
            dice_by_attr_cup[attr_label].append(metric_i[1][0])
            hd_by_attr_cup[attr_label].append(metric_i[1][1])
            jc_by_attr_cup[attr_label].append(metric_i[1][2])
            
            NSD_by_attr_cup[attr_label].append(nsd_metric[1])
            ASD_by_attr_cup[attr_label].append(asd_metric[1])

    
    metric_list = metric_list / len(db_test)
    mean_auc = auc_overall / len(db_test)
    mean_cdr_dist = cdr_overall / len(db_test) 
    
    nsd_list = nsd_list / len(db_test)
    asd_list = asd_list / len(db_test)
    
    # print(metric_list.shape)
    
    performance = np.mean(metric_list, axis=0)[0]
    mean_nsd = np.mean(nsd_list)
    mean_asd = np.mean(asd_list)

    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_jaccard = np.mean(metric_list, axis=0)[2]
    cup_overall_dice = metric_list[1][0]
    rim_overall_dice = metric_list[0][0]
    
    rim_overall_nsd = nsd_list[0][0][0]
    cup_overall_nsd = nsd_list[1][0][0]
    
    rim_overall_asd = asd_list[0]
    cup_overall_asd = asd_list[1]
    
    cup_overall_hd95 = metric_list[1][1]
    rim_overall_hd95 = metric_list[0][1]

    cup_overall_jaccard = metric_list[1][2]
    rim_overall_jaccard = metric_list[0][2]
    
    
    logging.info('--------- Overall Performance for Attribute: {} -----------'.format(args.attribute))

    for one_attr in range(no_of_attr):
        one_attr_dice_list = dice_by_attr[one_attr]
        one_attr_hd_list = hd_by_attr[one_attr]
        one_attr_jc_list = jc_by_attr[one_attr]
        one_attr_auc_list = auc_by_attr[one_attr]
        one_attr_cdr_list = cdr_by_attr[one_attr]
        
        one_attr_nsd_list = NSD_by_attr[one_attr]
        one_attr_asd_list = ASD_by_attr[one_attr]

        logging.info(f'{one_attr}-attr overall dice: {np.mean(one_attr_dice_list):.4f}')
        logging.info(f'{one_attr}-attr overall hd95: {np.mean(one_attr_hd_list):.4f}')
        logging.info(f'{one_attr}-attr overall Jaccard/IoU: {np.mean(one_attr_jc_list):.4f}')

        logging.info(f'{one_attr}-attr overall NSD: {np.mean(one_attr_nsd_list):.4f}')
        logging.info(f'{one_attr}-attr overall ASD: {np.mean(one_attr_asd_list):.4f}')

    logging.info('--------- Cup Performance for Attribute: {} -----------'.format(args.attribute))
        
    es_cup_dice = equity_scaled_perf(dice_by_attr_cup, cup_overall_dice, no_of_attr)
    es_cup_iou = equity_scaled_perf(jc_by_attr_cup, cup_overall_jaccard,no_of_attr)
   
    logging.info('----------- Cup %s - All -----------' % (args.attribute))
    logging.info('%.3f & %.3f & %.3f & %.3f' % (np.mean(es_cup_dice), np.mean(cup_overall_dice), np.mean(es_cup_iou), np.mean(cup_overall_jaccard)))
    logging.info('----------- Cup %s - All -----------' % (args.attribute))
       
    pred_cup_array = np.concatenate(all_preds_cup).flatten()
    gts_cup_array = np.concatenate(all_gts_cup).flatten()
    attr_cup_array = np.concatenate(all_attrs_cup).flatten()

    for one_attr in range(no_of_attr):
        one_attr_dice_list = dice_by_attr_cup[one_attr]
        one_attr_hd_list = hd_by_attr_cup[one_attr]
        one_attr_jc_list = jc_by_attr_cup[one_attr]
      
        one_attr_nsd_list = NSD_by_attr_cup[one_attr]
        one_attr_asd_list = ASD_by_attr_cup[one_attr]

        logging.info('----------- Cup %s - %d (n=%d)-----------' % (args.attribute, one_attr, len(one_attr_dice_list)))
        logging.info('%.3f & %.3f' % (np.mean(one_attr_dice_list), np.mean(one_attr_jc_list)) )
        logging.info('----------- Cup %s - %d (n=%d) -----------' % (args.attribute, one_attr, len(one_attr_dice_list)))


    logging.info('--------- Rim Performance for Attribute: {} -----------'.format(args.attribute))

    es_rim_dice = equity_scaled_perf(dice_by_attr_rim, rim_overall_dice, no_of_attr)
    es_rim_iou = equity_scaled_perf(jc_by_attr_rim, rim_overall_jaccard, no_of_attr)

    logging.info('\n----------- Rim %s - All -----------' % (args.attribute))
    logging.info('%.3f & %.3f & %.3f & %.3f' % (np.mean(es_rim_dice), np.mean(rim_overall_dice), np.mean(es_rim_iou), np.mean(rim_overall_jaccard)))
    logging.info('\n----------- Rim %s - All -----------' % (args.attribute))

    pred_rim_array = np.concatenate(all_preds_rim).flatten()
    gts_rim_array = np.concatenate(all_gts_rim).flatten()
    attr_rim_array = np.concatenate(all_attrs_rim).flatten()
    
    for one_attr in range(no_of_attr):
        one_attr_dice_list = dice_by_attr_rim[one_attr]
        one_attr_hd_list = hd_by_attr_rim[one_attr]
        one_attr_jc_list = jc_by_attr_rim[one_attr]

        one_attr_nsd_list = NSD_by_attr_rim[one_attr]
        one_attr_asd_list = ASD_by_attr_rim[one_attr]

        logging.info('\n----------- Rim %s - %d (n=%d) -----------' % (args.attribute, one_attr, len(one_attr_dice_list)))
        logging.info('%.3f & %.3f' % (np.mean(one_attr_dice_list), np.mean(one_attr_jc_list)) )
        logging.info('\n----------- Rim %s - %d (n=%d)  -----------' % (args.attribute, one_attr, len(one_attr_dice_list)))

    logging.info('------------------------------------------------------')
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jaccard : %f, mean_auc : %f, mean_cdr_distance : %f ' % (performance, mean_hd95, mean_jaccard, mean_auc, mean_cdr_dist))
    logging.info("Testing Finished!")

    return 1