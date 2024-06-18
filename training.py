"""
Adapted form MONAI Tutorial: https://github.com/Project-MONAI/tutorials/tree/main/2d_segmentation/torch
"""

import argparse
from contextlib import AbstractAsyncContextManager
import os
join = os.path.join
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import Logger, dice_coeff, compute_miou, PolyOptimizer, PolySGDOptimizer
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
from models.unet_pp import UNet_pp
from models.discriminator import FCDiscriminator
from monai.visualize import plot_2d_or_3d_image
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import random

from dataset import Dataset, create_interior_map

monai.config.print_config()
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print('Successfully import all requirements!')

def main():
    parser = argparse.ArgumentParser('Baseline for Microscopy image segmentation', add_help=False)
    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='training data path; subfolders: images, labels')
    parser.add_argument('--work_dir', default='debug',
                        help='path where to save models and logs')
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--resume', default=False,
                        help='resume from checkpoint')
    parser.add_argument('--num_workers', default=4, type=int)

    # Model parameters
    parser.add_argument('--model_name', default='swinunetr', help='select mode: unet, unetr, swinunetr')
    parser.add_argument('--num_class', default=4, type=int, help='segmentation classes')
    parser.add_argument('--input_size', default=480, type=int, help='segmentation classes')

    # Training parameters
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size per GPU')
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--val_interval', default=10, type=int) 
    parser.add_argument('--epoch_tolerance', default=220, type=int)
    parser.add_argument('--initial_lr', type=float, default=6e-4, help='learning rate')
    parser.add_argument('--use_resize', type=int, default=-1, help='Using Resize')
    parser.add_argument('--use_semi_cp', type=int, default=-1, help='Using Copy Paste for AF')
    parser.add_argument('--use_cp', type=int, default=-1, help='Using Copy Paste for RF')
    parser.add_argument('--semi_start', type=int, default=0, help='Epoch to Start SSL')
    
    parser.add_argument('--source', type=str, default="AF", help='source training set')
    parser.add_argument('--target', type=str, default="RF", help='target validation set')
    parser.add_argument('--unlabel', type=str, default="RF_unlabel", help='target unlabel set')
    parser.add_argument('--mode', type=str, default="blood", help='choroid/blood/multi-task')

    parser.add_argument('--rate_semi', type=float, default=1, help='semi loss')
    parser.add_argument('--rate_da', type=float, default=1, help='domain loss')
    parser.add_argument('--thresh_semi', type=float, default=0.85, help='confidence threshold of semi')
    parser.add_argument('--thresh_da', type=float, default=0.8, help='confidence threshold of semi')
    parser.add_argument('--initial_lr_d', type=float, default=6e-5, help='learning rate')
    parser.add_argument('--rate_multi', type=float, default=1, help='learning rate')
    parser.add_argument('--semi_loss', type=str, default="Dice", help='learning rate')


    # def main():
    #     args = parser.parse_args()
    args = parser.parse_args()

    #%% set training/validation split
    #np.random.seed(args.seed)
    #args.work_dir = os.path.join("train_logs", args.work_dir)

    seed = args.seed
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    monai.utils.set_determinism(seed=seed, additional_settings=None)
    print(args.seed)

    model_path = args.work_dir
    os.makedirs(model_path, exist_ok=True)

    vis_path = os.path.join(model_path, "vis")
    os.makedirs(vis_path, exist_ok=True)

    sys.stdout = Logger(model_path + "/log.txt")

    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(__file__, join(model_path, run_id + '_' + os.path.basename(__file__)))


    ##Training set
    img_path = join(args.data_path, args.source, "imgs")
    gt_blood_path = join(args.data_path, args.source, "bloods")
    gt_choroid_path = join(args.data_path, args.source, "choroids")
    img_names = sorted(os.listdir(img_path))
    gt_blood_names = [img_name.split('.')[0]+'.png' for img_name in img_names]
    gt_choroid_names = [img_name.split('.')[0]+'.png' for img_name in img_names]
    img_num = len(img_names)
    train_indices = np.arange(img_num)
    np.random.shuffle(train_indices)
    ##val set
    val_img_path = join(args.data_path, args.target, "imgs")
    val_gt_blood_path = join(args.data_path, args.target, "bloods")
    val_gt_choroid_path = join(args.data_path, args.target, "choroids")
    
    val_img_names = sorted(os.listdir(val_img_path))
    val_gt_blood_names = [val_img_name.split('.')[0]+'.png' for val_img_name in val_img_names]
    val_gt_choroid_names = [val_img_name.split('.')[0]+'.png' for val_img_name in val_img_names]

    val_img_num = len(val_img_names)
    val_indices = np.arange(val_img_num)
    train_files = [{"img": join(img_path, img_names[i]), "label_blood": join(gt_blood_path, gt_blood_names[i]), "label_choroid": join(gt_choroid_path, gt_choroid_names[i])} for i in train_indices]
    val_files = [{"img": join(val_img_path, val_img_names[i]), "label_blood": join(val_gt_blood_path, val_gt_blood_names[i]), "label_choroid": join(val_gt_choroid_path, val_gt_choroid_names[i])} for i in val_indices]

    #% define dataset, data loader
    print(f"training image num: {len(train_files)}, validation image num: {len(val_files)}")
    
    if args.rate_semi != 0:
        unsup_path = join(args.data_path, args.unlabel, "imgs")
        unsup_img_names = sorted(os.listdir(unsup_path))
        unsup_num = len(unsup_img_names)
        unsup_indices = np.arange(unsup_num)
        unsup_files = [{"img": join(unsup_path, unsup_img_names[i])} for i in unsup_indices]
        num_semi_files = np.int64(len(unsup_files))
        unsup_files = unsup_files[:num_semi_files]
        train_ds = Dataset(data=train_files, input_size=args.input_size, use_cp=args.use_cp, use_resize=args.use_resize)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        unsup_ds = Dataset(data=unsup_files, mode="semi", input_size=args.input_size, use_cp=args.use_cp, use_resize=args.use_resize)
        unsup_loader = DataLoader(
            unsup_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        data_loader_unsup_iter = enumerate(unsup_loader)

        print("Using Semi Supervised Training... Labeled: %d, UnLabeled %d"%(len(train_files), len(unsup_files)))
    else:
        train_ds = Dataset(data=train_files, input_size=args.input_size)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    # create a validation data loader
    val_ds = Dataset(data=val_files, mode="val", input_size=args.input_size)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    dice_metric_blood = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    dice_metric_choroid = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "multi-task":
        args.num_class = 5 #BG&HL&CL,BG&CV
    elif args.mode == "blood":
        args.num_class = 2
    else:
        args.num_class = 3

    if args.model_name.lower() == 'unet':
        model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=args.num_class,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)
        if args.semi_percent != 0:
            model_teacher = monai.networks.nets.UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=args.num_class,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                ).to(device)
            for p in model_teacher.parameters():
                p.requires_grad = False

    if args.model_name.lower() == 'swinunetr':
        model = monai.networks.nets.SwinUNETR(
            img_size=(args.input_size, args.input_size), 
            in_channels=1, 
            out_channels=args.num_class,
            feature_size=24, # should be divisible by 12
            spatial_dims=2
            ).to(device)
        if args.semi_percent != 0:
            model_teacher = monai.networks.nets.SwinUNETR(
                img_size=(args.input_size, args.input_size), 
                in_channels=1, 
                out_channels=args.num_class,
                feature_size=24, # should be divisible by 12
                spatial_dims=2
                ).to(device)
            for p in model_teacher.parameters():
                p.requires_grad = False


    if args.model_name.lower() == 'cunet':
        model = CU_Net().to(device)
        if args.rate_semi != 0:
            model_teacher = CU_Net().to(device)
            for p in model_teacher.parameters():
                p.requires_grad = False
    
    if args.model_name.lower() == 'attunet':
        model = AttU_Net(output_ch=args.num_class).to(device)
        if args.rate_semi != 0:
            model_teacher = AttU_Net(output_ch=args.num_class).to(device)
            for p in model_teacher.parameters():
                p.requires_grad = False

    if args.model_name.lower() == 'unet++':
        model = UNet_pp(out_channel=args.num_class).to(device)
        if args.rate_semi != 0:
            model_teacher = UNet_pp(out_channel=args.num_class).to(device)
            for p in model_teacher.parameters():
                p.requires_grad = False

    print(model)
    loss_function = monai.losses.DiceCELoss(softmax=True)
    initial_lr = args.initial_lr

    ##SGD for Sub-task Segmentor
    optimizer = torch.optim.AdamW(model.parameters(), initial_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150], gamma=0.1, verbose=False)

    ##SGD for EDD Module
    discriminator = FCDiscriminator(num_classes=args.num_class, ndf=64).cuda()
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), args.initial_lr_d)

    # start a typical PyTorch training
    max_epochs = args.max_epochs
    epoch_tolerance = args.epoch_tolerance
    val_interval = args.val_interval
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter(model_path)
    for epoch in range(1, max_epochs):
        model.train()
        epoch_loss = 0
        epoch_loss_d = 0
        epoch_loss_semi = 0
        epoch_loss_sup = 0
        for step, batch_data in enumerate(train_loader, 1):
            i_iter = (epoch-1) * len(train_loader) + step
            if args.mode == "blood":
                inputs, labels = batch_data["img"].to(device), batch_data["label_blood"].to(device)
                if args.num_class == 2:
                    labels[labels != 0] = 1
                labels_onehot = monai.networks.one_hot(labels, args.num_class) # (b,cls,256,256)
            elif args.mode == "choroid":
                inputs, labels = batch_data["img"].to(device), batch_data["label_choroid"].to(device)
                if args.num_class == 2:
                    labels[labels != 0] = 1
                labels_onehot = monai.networks.one_hot(labels, args.num_class) # (b,cls,256,256)
            elif args.mode == "multi-task":
                inputs, labels_blood, labels_choroid = batch_data["img"].to(device), batch_data["label_blood"].to(device), batch_data["label_choroid"].to(device)
                labels_blood[labels_blood != 0] = 1
                labels_onehot_choroid = monai.networks.one_hot(labels_choroid, 3) # (b,3,256,256)
                labels_onehot_blood = monai.networks.one_hot(labels_blood, 2) # (b,2,256,256)
                labels_onehot = torch.cat([labels_onehot_choroid, labels_onehot_blood], dim=1)
            if args.rate_semi != 0: ###Using un-labaled RF B-SCans
                ###Read Unsup data
                unlabel_idx, bathch_data = data_loader_unsup_iter.__next__()
                batch_data_unsup = bathch_data
                if unlabel_idx == len(unsup_loader) - 1:
                    data_loader_unsup_iter = enumerate(unsup_loader)
                inputs_unsup_weak = batch_data_unsup["img"].to(device)
                inputs_unsup_strong = batch_data_unsup["img_strong"].to(device)

                ###Update the Segmentors
                for param in discriminator.parameters():
                    param.requires_grad = False
                for param in model.parameters():
                    param.requires_grad = True

                ####Student Forward to Genertate Prediction of AF B-Scan
                outputs_sup = model(inputs)
                if args.mode == "multi-task": #Using CUNet Loss
                    loss_choroid = loss_function(outputs_sup[:, :3, :, :], labels_onehot[:, :3, :, :])
                    loss_blood = loss_function(outputs_sup[:, 3:, :, :], labels_onehot[:, 3:, :, :])
                    loss_student = loss_choroid + loss_blood + args.rate_multi * torch.abs(loss_blood.clone() - loss_choroid.clone())
                else:
                    loss_student = loss_function(outputs_sup, labels_onehot)

                ####Teacher forward to Genertate Prediction of Unlabaled RF B-Scan
                with torch.no_grad():
                    outputs_teacher_unsup = model_teacher(inputs_unsup_weak).detach()
                
                ####Generate Pseudo Annotation for Unlabaled RF B-SCans
                if args.mode == "multi-task":
                    conf_teacher_unsup_choroid = torch.softmax(outputs_teacher_unsup[:, :3, :, :], dim=1)
                    conf_teacher_unsup_blood = torch.softmax(outputs_teacher_unsup[:, 3:, :, :], dim=1)
                    select_unsup_choroid, gt_teacher_unsup_choroid = torch.max(conf_teacher_unsup_choroid, dim=1)
                    select_unsup_blood, gt_teacher_unsup_blood = torch.max(conf_teacher_unsup_blood, dim=1)
                    select_unsup = torch.cat([select_unsup_choroid.unsqueeze(1), select_unsup_blood.unsqueeze(1)], dim=1)
                else:
                    conf_teacher_unsup = torch.softmax(outputs_teacher_unsup, dim=1)
                    select_unsup, gt_teacher_unsup = torch.max(conf_teacher_unsup, dim=1)

                ####Discriminator Forward to Geneate the Domain Scores for AF B-Scans and the Adversial Domain GT
                d_sup = discriminator(outputs_sup)
                d_label_sup = torch.FloatTensor(d_sup.data.size()).fill_(1).cuda() #Adversial Domain GT for AF
                ####Discriminator Forward to Geneate the Domain Scores for "Embeded" RF B-Scans
                d_unsup = discriminator(outputs_teacher_unsup)
                d_label_unsup = torch.FloatTensor(d_unsup.data.size()).fill_(0).cuda() #Adversial Domain GT for RF

                ####Domain Adaptation Loss
                loss_d = (torch.mean((d_sup - d_label_sup).abs() ** 2) + torch.mean((d_unsup - d_label_unsup).abs() ** 2)) / 2

                ####Filtering Samples based on Domain Scores
                select_domain_unsup = d_unsup.detach().clone()
                select_domain_unsup = F.interpolate(select_domain_unsup, size=[inputs.shape[2], inputs.shape[3]])
                select_domain_unsup[select_domain_unsup > args.thresh_da] = 0 #Filtering domain score > beta2
                select_domain_unsup[select_domain_unsup < (1 - args.thresh_da)] = 0 #Filtering domain score < beta1
                select_domain_unsup[select_domain_unsup != 0] = 1

                #####Filtering Representive Samples based on Predictions
                if args.mode == "multi-task":
                    select_unsup_choroid[select_unsup_choroid > args.thresh_semi] = 1
                    select_unsup_choroid[select_unsup_choroid != 1] = 0

                    select_unsup_blood[select_unsup_blood > args.thresh_semi] = 1
                    select_unsup_blood[select_unsup_blood != 1] = 0
                    if args.thresh_da != 0:
                        select_unsup_blood = select_unsup_blood * select_domain_unsup.squeeze(1)
                        select_unsup_choroid = select_unsup_choroid * select_domain_unsup.squeeze(1)
                    save_select = select_unsup_blood.clone() + select_unsup_choroid.clone()
                    save_select[save_select != 0] = 1

                    save_select_choroid = select_unsup_choroid.clone()
                    save_select_blood = select_unsup_blood.clone()

                    save_select = save_select_blood * save_select_choroid

                    select_unsup_choroid = select_unsup_choroid.view(-1).bool()
                    select_unsup_blood = select_unsup_blood.view(-1).bool()
                else:
                    select_unsup[select_unsup > args.thresh_semi] = 1
                    select_unsup[select_unsup != 1] = 0
                    if args.thresh_da != 0 and torch.sum(select_domain_unsup) != 0:
                        select_unsup = select_unsup * select_domain_unsup.squeeze(1)
                    save_select = select_unsup.clone()
                    select_unsup = select_unsup.view(-1).bool()

                ####Semi-supervised Loss
                outputs_unsup = model(inputs_unsup_strong)
                if args.mode == "multi-task":
                    gt_teacher_unsup_onehot_choroid = monai.networks.one_hot(gt_teacher_unsup_choroid.unsqueeze(1), 3) # (b,cls,256,256)
                    gt_teacher_unsup_onehot_blood = monai.networks.one_hot(gt_teacher_unsup_blood.unsqueeze(1), 2) # (b,cls,256,256)
                    gt_teacher_unsup_onehot = torch.cat([gt_teacher_unsup_onehot_choroid, gt_teacher_unsup_onehot_blood], dim=1)
                    loss_concist_choroid = loss_function(outputs_unsup[:, :3, :, :].permute(0, 2, 3, 1).contiguous().view(-1, 3)[select_unsup_choroid, :], gt_teacher_unsup_onehot[:, :3, :, :].permute(0, 2, 3, 1).contiguous().view(-1, 3)[select_unsup_choroid, :])
                    loss_concist_blood = loss_function(outputs_unsup[:, 3:, :, :].permute(0, 2, 3, 1).contiguous().view(-1, 2)[select_unsup_blood, :], gt_teacher_unsup_onehot[:, 3:, :, :].permute(0, 2, 3, 1).contiguous().view(-1, 2)[select_unsup_blood, :]) 
                    loss_concist = (args.rate_multi * loss_concist_blood + loss_concist_choroid) / 2
                else:
                    gt_teacher_unsup_onehot = monai.networks.one_hot(gt_teacher_unsup.unsqueeze(1), args.num_class) # (b,cls,256,256)
                    loss_concist = loss_function(outputs_unsup.permute(0, 2, 3, 1).contiguous().view(-1,  args.num_class)[select_unsup, :], gt_teacher_unsup_onehot.permute(0, 2, 3, 1).contiguous().view(-1, args.num_class)[select_unsup, :]) #loss_concist_1 + loss_concist_2
                
                ####Calculate Total Loss
                if epoch < args.semi_start:
                    loss = loss_student
                else:
                    loss = loss_student + args.rate_semi * loss_concist + args.rate_da * loss_d

                ####Update Segmentors with SGD
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ####Update Teacher
                ema_decay = min(1-1/(i_iter+1),0.99)
                for t_params, s_params in zip(model_teacher.parameters(), model.parameters()):
                    t_params.mul_(ema_decay).add_(1-ema_decay, s_params.data)
                
                ####Update Dicriminator with SGD
                if epoch >= args.semi_start:
                    for param in discriminator.parameters():
                        param.requires_grad = True
                    for param in model.parameters():
                        param.requires_grad = False
                    d_sup = discriminator(outputs_sup.detach())
                    d_label_sup = torch.FloatTensor(d_sup.data.size()).fill_(0).cuda()
                    d_unsup = discriminator(outputs_teacher_unsup.detach())
                    d_label_unsup = torch.FloatTensor(d_unsup.data.size()).fill_(1).cuda()
                    loss_d = (torch.mean((d_sup - d_label_sup).abs() ** 2) + torch.mean((d_unsup - d_label_unsup).abs() ** 2)) / 2

                    optimizer_d.zero_grad()
                    loss_d.backward()
                    optimizer_d.step()

                ####Visualization
                if step % 1 == 0:
                    save_img_unsup = inputs_unsup_strong[0].permute(1, 2, 0)[:, :, 0]
                    if args.mode == "multi-task":
                        save_teacher_choroid = gt_teacher_unsup_choroid[0]
                        save_teacher_blood = gt_teacher_unsup_blood[0]
                        save_student_choroid = torch.argmax(outputs_unsup[:, :3, :, :], dim=1)[0]
                        save_student_blood = torch.argmax(outputs_unsup[:, 3:, :, :], dim=1)[0]
                        save_unsup = torch.cat((inputs_unsup_weak[0].permute(1, 2, 0)[:, :, 0], save_img_unsup, save_select_choroid[0], save_select_blood[0], save_teacher_choroid, save_student_choroid, save_teacher_blood, save_student_blood, save_teacher_blood*save_teacher_choroid, save_student_blood*save_student_choroid), dim=1)
                        plt.imsave(os.path.join(vis_path, "%d_unsup.png")%(step), save_unsup.cpu().numpy())
                    else:
                        save_teacher = gt_teacher_unsup[0]#.expand(save_img_unsup.shape) #/ args.num_class
                        save_student = torch.argmax(outputs_unsup, dim=1)[0]#.expand(save_img_unsup.shape) #/ args.num_class
                        save_unsup = torch.cat((inputs_unsup_weak[0].permute(1, 2, 0)[:, :, 0], save_img_unsup, save_select[0], save_teacher, save_student), dim=1)
                        plt.imsave(os.path.join(vis_path, "%d_unsup.png")%(step), save_unsup.cpu().numpy())

                    import cv2
                    #cv2.imwrite(os.path.join(vis_path, "%d_img_sup.jpg")%(step), np.uint8(inputs[0, 0].detach().cpu() *255.0))
                    plt.imsave(os.path.join(vis_path, "%d_img_sup.jpg")%(step), inputs[0, 0].detach().cpu())
                    cv2.imwrite(os.path.join(vis_path, "%d_img.jpg")%(step), np.uint8(inputs_unsup_weak[0, 0].detach().cpu() *255.0))
                    plt.imsave(os.path.join(vis_path, "%d_select.png")%(step), save_select[0].detach().cpu())
                    save_domain = torch.cat((d_sup[0, 0].detach().cpu(), d_unsup[0, 0].detach().cpu()), dim=1)
                    plt.imsave(os.path.join(vis_path, "%d_domain.png")%(step), save_domain.numpy())

                epoch_loss += loss.item()
                epoch_loss_sup += loss_student.item()
                epoch_loss_d += loss_d.item()
                epoch_loss_semi += loss_concist.item()
            else: ###Only Supervised Training
                outputs_sup = model(inputs)
                if args.mode == "multi-task":
                    loss = loss_function(outputs_sup[:, :3, :, :], labels_onehot[:, :3, :, :]) + loss_function(outputs_sup[:, 3:, :, :], labels_onehot[:, 3:, :, :])
                else:
                    loss = loss_function(outputs_sup, labels_onehot)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_loss_sup += 0
                epoch_loss_d += 0
                epoch_loss_semi += 0
    
        epoch_loss /= step
        epoch_loss_sup /= step
        epoch_loss_d /= step
        epoch_loss_semi /= step
        epoch_loss_values.append(epoch_loss)

        epoch_len = len(train_ds) // train_loader.batch_size
        writer.add_scalar("total_loss", epoch_loss, epoch)
        writer.add_scalar("supervised_loss", epoch_loss_sup, epoch)
        writer.add_scalar("discriminator_loss", epoch_loss_d, epoch)
        writer.add_scalar("unsupervised_loss", epoch_loss_semi, epoch)

        print(f"epoch {epoch} average loss: {epoch_loss:.4f} loss_d: {epoch_loss_d:.4f} loss_semi: {epoch_loss_semi:.4f}")
        checkpoint = {'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss_values,
                }
        scheduler.step()

        if epoch>20 and epoch % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for i, val_data in enumerate(val_loader):
                    if args.mode == "multi-task":
                        val_images, val_labels_blood, val_labels_choroid = val_data["img"].to(device), val_data["label_blood"].to(device), val_data["label_choroid"].to(device)
                        val_labels_blood[val_labels_blood!=0] = 1
                        val_labels_onehot_choroid = monai.networks.one_hot(val_labels_choroid, 3) # (b,4,256,256)
                        val_labels_onehot_blood = monai.networks.one_hot(val_labels_blood, 2) # (b,2,256,256)
                        roi_size = (args.input_size, args.input_size)
                        sw_batch_size = 4
                        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                        val_outputs_choroid = torch.softmax(val_outputs[:, :3, :, :], dim=1)
                        val_outputs_choroid = torch.argmax(val_outputs_choroid, dim=1).unsqueeze(1)
                        val_outputs_one_hot_choroid = monai.networks.one_hot(val_outputs_choroid, 3)

                        val_outputs_blood = torch.softmax(val_outputs[:, 3:, :, :], dim=1)
                        val_outputs_blood = torch.argmax(val_outputs_blood, dim=1).unsqueeze(1)
                        val_outputs_one_hot_blood = monai.networks.one_hot(val_outputs_blood, 2)

                        dice_metric_blood(y_pred=val_outputs_one_hot_blood, y=val_labels_onehot_blood)
                        dice_metric_choroid(y_pred=val_outputs_one_hot_choroid, y=val_labels_onehot_choroid)

                        if i % 5 == 0:
                            save_img_sup = val_images[0].permute(1, 2, 0)[:, :, 0]
                            save_sup = torch.cat((save_img_sup,  val_labels_choroid[0, 0], val_outputs_choroid[0, 0], val_labels_blood[0, 0], val_outputs_blood[0, 0]), dim=1)
                            plt.imsave(os.path.join(vis_path, "%d_val.png")%(i), save_sup.cpu().numpy())
                    else:
                        if args.mode == "blood":
                            val_images, val_labels = val_data["img"].to(device), val_data["label_blood"].to(device)
                        elif args.mode == "choroid":
                            val_images, val_labels = val_data["img"].to(device), val_data["label_choroid"].to(device)
                        if args.num_class == 2:
                            val_labels[val_labels != 0] = 1
                        val_labels_onehot = monai.networks.one_hot(val_labels, args.num_class)
                        roi_size = (args.input_size, args.input_size)
                        sw_batch_size = 4
                        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                        val_outputs = torch.softmax(val_outputs, dim=1)
                        val_outputs = torch.argmax(val_outputs, dim=1).unsqueeze(1)
                        val_outputs_one_hot = monai.networks.one_hot(val_outputs, args.num_class)
                        if i % 5 == 0:
                            save_img_sup = val_images[0].permute(1, 2, 0)[:, :, 0]
                            save_sup = torch.cat((save_img_sup,  val_labels[0, 0], val_outputs[0, 0]), dim=1)
                            plt.imsave(os.path.join(vis_path, "%d_val.png")%(i), save_sup.cpu().numpy())
                        dice_metric(y_pred=val_outputs_one_hot, y=val_labels_onehot)

                # aggregate the final mean dice result
                if args.mode == "multi-task":
                    metric_blood = dice_metric_blood.aggregate()
                    metric_blood = torch.mean(metric_blood).item()
                    metric_choroid = dice_metric_choroid.aggregate()
                    metric_choroid = torch.mean(metric_choroid).item()
                    metric = (metric_blood + metric_choroid) / 2
                    writer.add_scalar("val/mean_dice", metric, epoch + 1)
                    writer.add_scalar("val/blood_dice", metric_blood, epoch + 1)
                    writer.add_scalar("val/choroid_dice", metric_choroid, epoch + 1)
                else:
                    metric = dice_metric.aggregate()
                    metric = torch.mean(metric).item()
                    if args.mode == "blood":
                        writer.add_scalar("val/blood_dice", metric, epoch + 1)
                    elif args.mode == "choroid":
                        writer.add_scalar("val/choroid_dice", metric, epoch + 1)


                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(checkpoint, join(model_path, "best_Dice_model.pth"))
                    print("saved new best metric model")
                    
                # reset the status for next validation round
                if args.mode == "multi-task":
                    dice_metric_blood.reset()
                    dice_metric_choroid.reset()
                    print(
                        "current epoch: {} choroid dice: {:.4f}, blood dice: {:.4f} mean dice: {:.4f}, best dice: {:.4f} at epoch {}".format(
                            epoch + 1, metric_choroid, metric_blood, metric, best_metric, best_metric_epoch
                        )
                    )
                else:
                    dice_metric.reset()
                    print(
                        "current epoch: {} current mean dice: {:.4f}, best dice: {:.4f} at epoch {}".format(
                            epoch + 1, metric, best_metric, best_metric_epoch
                        )
                    )

            if (epoch - best_metric_epoch) > epoch_tolerance:
                print(f"validation metric does not improve for {epoch_tolerance} epochs! current {epoch}, {best_metric_epoch}")
                break

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
    torch.save(checkpoint, join(model_path, 'final_model.pth'))
    np.savez_compressed(join(model_path, 'train_log.npz'), val_dice=metric_values, epoch_loss=epoch_loss_values)


if __name__ == "__main__":
    main()
