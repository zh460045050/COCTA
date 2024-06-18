

import argparse
from contextlib import AbstractAsyncContextManager
import os
join = os.path.join
import cv2
import numpy as np
import torch
from skimage.segmentation import mark_boundaries
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
from models.attunet import AttU_Net
from models.msunet import CU_Net
from models.discriminator import FCDiscriminator
import matplotlib.pyplot as plt
from datetime import datetime

from dataset import Dataset

join = os.path.join
def main():
    parser = argparse.ArgumentParser('Baseline for Microscopy image segmentation', add_help=False)
    parser.add_argument('--subject_dir', default='', type=str,
                        help='training data path; subfolders: images, labels')
    parser.add_argument('--out_dir', default='', type=str,
                        help='training data path; subfolders: images, labels')
    parser.add_argument('--check_name', default='debug', type=str,
                        help='training data path; subfolders: images, labels')
    parser.add_argument('--model_name', default='swinunetr', type=str,
                        help='training data path; subfolders: images, labels')
    parser.add_argument('--check_path_blood', default='', type=str,
                        help='training data path; subfolders: images, labels')
    parser.add_argument('--check_path_choroid', default='', type=str,
                        help='training data path; subfolders: images, labels')
    args = parser.parse_args()

    args.input_size = 480
    args.save_dir = args.out_dir
    os.makedirs(args.save_dir, exist_ok=True)

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
            out_channels=3,
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
            out_channels=3,
            feature_size=24, # should be divisible by 12
            spatial_dims=2
            ).to(device)

    if args.model_name.lower() == 'cunet':
        model = CU_Net().to(device)
    
    if args.model_name.lower() == 'attunet':
        model = AttU_Net(output_ch=2).to(device)
        model_choroid = AttU_Net(output_ch=3).to(device)

    if args.model_name.lower() == 'unet++':
        from models.unet_pp import UNet_pp
        model = UNet_pp(out_channel=2).to(device)
        model_choroid = UNet_pp(out_channel=3).to(device)


    model.load_state_dict(torch.load(args.check_path_blood)['model_state_dict'], strict=True)
    model.eval()

    if args.model_name.lower() != 'cunet':
        model_choroid.load_state_dict(torch.load(args.check_path_choroid)['model_state_dict'], strict=True)
        model_choroid.eval()

    persons = os.listdir(args.subject_dir)
    for b, person in enumerate(persons):
        print("%d/%d"%(b, len(persons)))
        def func(info):
            return int(info[:-4])
        val_img_names = sorted(os.listdir(join(args.subject_dir, person)), key=func)
        val_img_path = join(args.subject_dir, person)
        val_img_num = len(val_img_names)
        val_indices = np.arange(val_img_num)
        val_files = [{"img": join(val_img_path, val_img_names[i])} for i in val_indices]
        val_ds = Dataset(data=val_files, mode="val", input_size=args.input_size)
        #val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=1)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
        bscan_dir = os.path.join(args.save_dir, "BScans", person)
        os.makedirs(bscan_dir, exist_ok=True)
        count = 0
        for i, val_data in enumerate(val_loader):
            val_images = val_data["img"].to(device)
            roi_size = (args.input_size, args.input_size)
            #sw_batch_size = 64
            sw_batch_size = 4
            with torch.no_grad():
                if args.model_name.lower() != 'cunet':
                    val_outputs_blood = sliding_window_inference(val_images, roi_size, sw_batch_size, model, overlap=0.5, mode="gaussian")
                    val_outputs_choroid = sliding_window_inference(val_images, roi_size, sw_batch_size, model_choroid, overlap=0.5, mode="gaussian")
                else:
                    val_output = sliding_window_inference(val_images, roi_size, sw_batch_size, model, overlap=0.5, mode="gaussian")
                    val_outputs_blood = val_output[:, 3:, :, :]
                    val_outputs_choroid = val_output[:, :3, :, :]
            val_outputs_blood = torch.softmax(val_outputs_blood, dim=1)
            val_outputs_blood = torch.argmax(val_outputs_blood, dim=1)

            val_outputs_choroid = torch.softmax(val_outputs_choroid, dim=1)
            val_outputs_choroid = torch.argmax(val_outputs_choroid, dim=1)

            val_outputs_blood_multi = val_outputs_choroid * val_outputs_blood

            
            if i == 0:
                for b in range(0, val_outputs_blood_multi.shape[0]):
                    val_image = val_images[b, 0]
                    save_bscans = mark_boundaries( np.uint8(val_image.cpu().data * 255.0), np.int64(val_outputs_blood_multi[b].cpu().data))
                    count = count + 1
                    plt.imsave(os.path.join(bscan_dir, "%d.jpg"%(count)), save_bscans)
            

            val_outputs_one_hot_blood_multi = monai.networks.one_hot(val_outputs_blood_multi.unsqueeze(1), 4)
            val_outputs_one_hot_choroid_multi = monai.networks.one_hot(val_outputs_choroid.unsqueeze(1), 4)

            if i == 0:
                density_map_multi = torch.sum(val_outputs_one_hot_blood_multi, dim=2).cpu()
                thickness_multi = torch.sum(val_outputs_one_hot_choroid_multi, dim=2).cpu()
            else:
                density_map_multi = torch.cat([density_map_multi, torch.sum(val_outputs_one_hot_blood_multi, dim=2).cpu()], dim=0)
                thickness_multi = torch.cat([thickness_multi, torch.sum(val_outputs_one_hot_choroid_multi, dim=2).cpu()], dim=0)

        all_layer = density_map_multi[:, 1, :] + density_map_multi[:, 2, :]
        min_layer, _ = torch.min(all_layer.view(-1), dim=0)
        max_layer, _ = torch.max(all_layer.view(-1), dim=0)
        all_layer = (all_layer - min_layer.unsqueeze(0).unsqueeze(-1)) / (max_layer.unsqueeze(0).unsqueeze(-1) - min_layer.unsqueeze(0).unsqueeze(-1))
        all_layer = np.uint8(np.array(all_layer.cpu().data * 255))


        min_layer, _ = torch.min(density_map_multi.permute(0, 2, 1).contiguous().view(-1, 4), dim=0)
        max_layer, _ = torch.max(density_map_multi.permute(0, 2, 1).contiguous().view(-1, 4), dim=0)
        density_map_multi = (density_map_multi - min_layer.unsqueeze(0).unsqueeze(-1)) / (max_layer.unsqueeze(0).unsqueeze(-1) - min_layer.unsqueeze(0).unsqueeze(-1))
        density_map_multi = np.uint8(np.array(density_map_multi.cpu().data * 255))

        os.makedirs(args.save_dir, exist_ok=True)

        cv2.imwrite(os.path.join(args.save_dir, "%s_SV.jpg"%(person)), density_map_multi[:, 1, :])
        cv2.imwrite(os.path.join(args.save_dir, "%s_HV.jpg"%(person)), density_map_multi[:, 2, :])
        cv2.imwrite(os.path.join(args.save_dir, "%s_CV.jpg"%(person)), all_layer)


        #cv2.imwrite(os.path.join(thick_dir, "%s_mid_%s_ct.jpg"%(person, args.model_name)), np.int64(thickness_multi[:, 1, :].cpu().data))
        #cv2.imwrite(os.path.join(thick_dir, "%s_big_%s_ct.jpg"%(person, args.model_name)), thickness_multi[:, 2, :].cpu().data)

        #cv2.imwrite(os.path.join(enface_dir, "%s_mid_%s_enface.jpg"%(person, args.model_name)), enface_multi[:, 1, :].cpu().data)
        #cv2.imwrite(os.path.join(enface_dir, "%s_big_%s_enface.jpg"%(person, args.model_name)), enface_multi[:, 2, :].cpu().data)
        #cv2.imwrite(os.path.join(args.save_dir, "%s_mid_%s.jpg"%(person, args.model_name)), save_map)
        #cv2.imwrite(os.path.join(args.save_dir, "%s_big_%s.jpg"%(person, args.model_name)), save_map)


if __name__ == "__main__":
    main()
