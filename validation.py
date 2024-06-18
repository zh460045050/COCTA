
import argparse
from contextlib import AbstractAsyncContextManager
import os
join = os.path.join
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import Logger, dice_coeff, compute_miou
import sys
from skimage import io, segmentation, morphology, measure, exposure
import monai
from monai.data import decollate_batch, PILReader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from stardist.matching import matching
from utils import PathIndex
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.attunet import AttU_Net
from models.msunet import CU_Net
from monai.visualize import plot_2d_or_3d_image
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import random
import cv2

from dataset import Dataset, create_interior_map

monai.config.print_config()
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print('Successfully import all requirements!')


def main():
    parser = argparse.ArgumentParser('Baseline for Microscopy image segmentation', add_help=False)
    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='training data path; subfolders: images, labels')
    parser.add_argument('--work_dir', default='test_logs',
                        help='path where to save models and logs')
    parser.add_argument('--num_workers', default=4, type=int)

    # Model parameters
    parser.add_argument('--model_name', default='swinunetr', help='select mode: unet, unetr, swinunetr')
    parser.add_argument('--input_size', default=480, type=int, help='segmentation classes')

    # Training parameters
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size per GPU')
    parser.add_argument('--check_path_blood', default="", type=str, help='Check Path of Blood Sub-Segmentor')
    parser.add_argument('--check_path_choroid', default="", type=str, help='Check Path of Layer Sub-Segmentor')

    parser.add_argument('--target', type=str, default="RF", help='test set')
    

    # def main():
    #     args = parser.parse_args()
    args = parser.parse_args()

    #%% set training/validation split
    #np.random.seed(args.seed)
    #args.work_dir = os.path.join("train_logs", args.work_dir)

    model_path = args.work_dir
    os.makedirs(model_path, exist_ok=True)

    vis_path = os.path.join(model_path, "vis")
    os.makedirs(vis_path, exist_ok=True)

    sys.stdout = Logger(model_path + "/log.txt")

    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(__file__, join(model_path, run_id + '_' + os.path.basename(__file__)))

    ##val set
    val_img_path = join(args.data_path, args.target, "imgs")
    val_gt_blood_path = join(args.data_path, args.target, "bloods")
    val_gt_choroid_path = join(args.data_path, args.target, "choroids")
    
    val_img_names = sorted(os.listdir(val_img_path))
    val_gt_blood_names = [val_img_name.split('.')[0]+'.png' for val_img_name in val_img_names]
    val_gt_choroid_names = [val_img_name.split('.')[0]+'.png' for val_img_name in val_img_names]

    val_img_num = len(val_img_names)
    val_indices = np.arange(val_img_num)
    val_files = [{"img": join(val_img_path, val_img_names[i]), "label_blood": join(val_gt_blood_path, val_gt_blood_names[i]), "label_choroid": join(val_gt_choroid_path, val_gt_choroid_names[i])} for i in val_indices]
    print(f"validation image num: {len(val_files)}")
    
    val_ds = Dataset(data=val_files, mode="val", input_size=args.input_size)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    dice_metric_blood = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
    dice_metric_choroid = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
    dice_metric_blood_multi = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_name.lower() == 'unet':
        model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)
        model_choroid = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)

    if args.model_name.lower() == 'swinunetr':
        model = monai.networks.nets.SwinUNETR(
            img_size=(args.input_size, args.input_size), 
            in_channels=1, 
            out_channels=2,
            feature_size=24, # should be divisible by 12
            spatial_dims=2
            ).to(device)

        model_choroid = monai.networks.nets.SwinUNETR(
            img_size=(args.input_size, args.input_size), 
            in_channels=1, 
            out_channels=4,
            feature_size=24, # should be divisible by 12
            spatial_dims=2
            ).to(device)

    if args.model_name.lower() == 'cunet':
        model = CU_Net().to(device)
    
    if args.model_name.lower() == 'attunet':
        model = AttU_Net(output_ch=2).to(device)
        model_choroid = AttU_Net(output_ch=4).to(device)

    if args.model_name.lower() == 'unet++':
        from models.unet_pp import UNet_pp
        model = UNet_pp(out_channel=2).to(device)
        model_choroid = UNet_pp(out_channel=4).to(device)

    model.load_state_dict(torch.load(args.check_path_blood)['model_state_dict'], strict=True)
    model.eval()
    if args.model_name.lower() != 'cunet':
        model_choroid.load_state_dict(torch.load(args.check_path_choroid)['model_state_dict'], strict=True)
        model_choroid.eval()

    with torch.no_grad():
        val_images = None
        for i, val_data in enumerate(val_loader):
            val_images, val_labels_blood, val_labels_choroid = val_data["img"].to(device), val_data["label_blood"].to(device), val_data["label_choroid"].to(device)
            val_labels_blood_multi = val_labels_blood.clone()
            val_labels_blood[val_labels_blood != 0] = 1
            val_labels_onehot_choroid = monai.networks.one_hot(val_labels_choroid, 4) # (b,4,256,256)
            val_labels_onehot_blood = monai.networks.one_hot(val_labels_blood, 2) # (b,2,256,256)
            val_labels_onehot_blood_multi = monai.networks.one_hot(val_labels_blood_multi, 4) # (b,2,256,256)

            roi_size = (args.input_size, args.input_size)
            sw_batch_size = 4

            if args.model_name.lower() == 'cunet':
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                val_outputs_choroid = val_outputs[:, :3, :, :]
                val_outputs_blood = val_outputs[:, 3:, :, :]
            else:
                val_outputs_blood = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                val_outputs_choroid = sliding_window_inference(val_images, roi_size, sw_batch_size, model_choroid)

            val_outputs_blood = torch.softmax(val_outputs_blood, dim=1)
            val_outputs_blood = torch.argmax(val_outputs_blood, dim=1).unsqueeze(1)
            val_outputs_one_hot_blood = monai.networks.one_hot(val_outputs_blood, 2)

            val_outputs_choroid = torch.softmax(val_outputs_choroid, dim=1)
            val_outputs_choroid = torch.argmax(val_outputs_choroid, dim=1).unsqueeze(1)
            val_outputs_one_hot_choroid = monai.networks.one_hot(val_outputs_choroid, 4)

            val_outputs_blood_multi = val_outputs_choroid * val_outputs_blood
            val_outputs_one_hot_blood_multi = monai.networks.one_hot(val_outputs_blood_multi, 4)
            
            dice_metric_blood(y_pred=val_outputs_one_hot_blood, y=val_labels_onehot_blood)
            dice_metric_blood_multi(y_pred=val_outputs_one_hot_blood_multi, y=val_labels_onehot_blood_multi)
            dice_metric_choroid(y_pred=val_outputs_one_hot_choroid, y=val_labels_onehot_choroid)

        metric_blood = dice_metric_blood.aggregate()
        metric_choroid = dice_metric_choroid.aggregate()
        metric_blood_multi = dice_metric_blood_multi.aggregate()

        print("CV: ", metric_blood)
        print("SL & HL:", metric_choroid)
        print("SV & HV:", metric_blood_multi)


            
if __name__ == "__main__":
    main()
