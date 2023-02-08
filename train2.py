import argparse
import copy
# from torch.utils.tensorboard import SummaryWriter
import csv
import os
import random
import time
from datetime import datetime
from pprint import pprint
from random import randint

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dl_src.dataset import CSVDataset, AttributesDataset, mean, std
from dl_src.model import MultiOutputModel
from test import calculate_metrics, validate


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch


def visualize_grid(model, dataloader, attr, device, show_cn_matrices=True, checkpoint=None, csv_filename=None,
                   caption=''):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)
    model.eval()
    gt_all = {x: list() for x in attr.fld_names}
    predicted_all = {x: list() for x in attr.fld_names}
    accuracyes = {x: 0 for x in attr.fld_names}

    list_of_images_to_check = []
    loitc_columns = ['filename', 'choice']

    with torch.no_grad():
        for batch in dataloader:
            img = batch['img'].to(device)
            gt_s = {x: batch['labels'][x] for x in attr.fld_names}
            fnames = batch['img_path']
            output = model(img)
            batches = calculate_metrics(output, gt_s)
            for x in batches:
                accuracyes[x] += batches[x]
            predicted_s = {x: output[x].cpu().max(1)[1] for x in attr.fld_names}

            for i in range(img.shape[0]):
                predicted = {x: attr.labels_id_to_name[x][predicted_s[x][i].item()]
                             for x in attr.fld_names}
                gt = {x: attr.labels_id_to_name[x][gt_s[x][i].item()] for x in attr.fld_names}
                for x in attr.fld_names:
                    gt_v = gt[x]
                    prdv = predicted[x]
                    gt_all[x].append(gt_v)
                    predicted_all[x].append(prdv)
                    if gt_v == prdv:
                        continue
                    list_of_images_to_check.append(
                        {loitc_columns[0]: os.path.split(fnames[i])[1],
                         loitc_columns[1]: f'<{gt_v}> or <{prdv}>'}
                    )

    if csv_filename is not None and list_of_images_to_check:
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=loitc_columns)
            writer.writeheader()
            writer.writerows(list_of_images_to_check)

    # Draw confusion matrices
    if show_cn_matrices:
        for x in attr.fld_names:
            cn_matrix = confusion_matrix(
                y_true=gt_all[x],
                y_pred=predicted_all[x],
                labels=attr.labels[x],
                # )
                normalize='true')
            ConfusionMatrixDisplay(cn_matrix, display_labels=attr.labels[x]).plot(
                include_values=True, xticks_rotation='vertical')
            plt.title(f"{x}:{caption}")
            plt.tight_layout()
            plt.show(block=False)

    model.train()


def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def checkpoint_save(model, name, epoch):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)
    return f


def cut_pil_image_(image: Image, border=20):
    w, h = image.size
    return image.crop((border, border, w - border, h - border))


def cut_pil_image__(image: Image, border=20, spread=.1):
    w, h = image.size
    s = spread * random.random()
    shift = int(w * s)
    return image.crop((border + shift, border + shift, w - border - shift, h - border - shift))


def cut_pil_image3(image: Image, border=20, spread=.1):
    w, h = image.size
    shiftx = border // 2 + int(border * random.random())
    shifty = border // 2 + int(border * random.random())
    w2, h2 = w - 2 * border, h - 2 * border
    return image.crop((shiftx, shifty, w2, h2))


def cut_pil_image(image: Image, border=20, spread=.5):
    w, h = image.size
    dw, dh = int(border * spread), int(border * spread)
    return image.crop((
        border,
        randint(border - dw, border + dh),
        w - border,
        randint(h - border - dh, h - border + dw)))


def cut_pil_image_v_old(image: Image, border=20, spread=.5):
    w, h = image.size
    dw, dh = int(border * spread), int(border * spread)
    return image.crop((
        randint(border - dw, border + dw),
        randint(border - dw, border + dh),
        randint(w - border - dw, w - border + dw),
        randint(h - border - dh, h - border + dw)))


def cut_pil_image_v_old(image: Image, border=20, spread=.05):
    w, h = image.size
    iw, ih = w - 2 * border, h - 2 * border
    dw, dh = int(iw * spread), int(ih * spread)
    return image.crop((
        randint(border // 2, border + dw),
        randint(border // 2, border + dh),
        randint(w - border - dw, w - border // 2),
        randint(h - border - dh, h - border // 2)))


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                f = os.path.join(savedir, f'{args.checkpoint_best}.pth')
                torch.save(model.state_dict(), f)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSV Training pipeline')
    parser.add_argument('--images_dir', type=str, required=True,
                        help="Folder containing images described in CSV file")
    parser.add_argument('--train_file', type=str, required=True,
                        help="CSV-file format (image name, label1, label2, ...) to use for training")
    parser.add_argument('--work_dir', type=str, required=True,
                        help="Folder to store trained model, logs. result etc")
    parser.add_argument('--attributes_file', type=str,
                        help="Path to the file with attributes. Must be set if train-file is a part "
                             "of a bigger file")
    parser.add_argument('--val_file', type=str, required=True,
                        help="Part of the dataset that will be used for training. Rest - for validation")
    parser.add_argument('--n_epochs', type=int, default=50, help="number of training epoch's")
    parser.add_argument('--batch_size', type=int, default=16, help="number of training epoch's")
    parser.add_argument('--num_workers', type=int, default=10, help="number of workers")
    parser.add_argument('--device', type=str, default='cuda', help="Device: 'cuda' or 'cpu'")
    parser.add_argument('--graph_out', type=str, default='true', help="true* / false")
    parser.add_argument('--checkpoint_best', type=str, default='checkpoint-best', help="checkpoint-best")
    args = parser.parse_args()
    print(f'[train] arguments')
    for k, arg in args.__dict__.items():
        print(f"{k:.<20}: {arg}")
    pprint(args.__dict__)
    TRAIN_PERIODE = 15
    COUNT_OF_LR_STEPS = 3

    start_epoch = 1
    N_epochs = args.n_epochs
    batch_size = args.batch_size
    num_workers = args.num_workers  # number of processes to handle dataset loading
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f'[DEVICE] {device}')
    if args.attributes_file is None:
        args.attributes_file = args.train_file

    attributes_file = os.path.join(args.work_dir, 'data.csv')
    print(f'[train] attributes: {attributes_file}')
    attributes = AttributesDataset(attributes_file)

    # specify image transforms for augmentation during training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Lambda(lambd=cut_pil_image),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # during validation we use only tensor and normalization transforms
    val_transform = transforms.Compose([
        transforms.Lambda(lambd=lambda x: cut_pil_image(x, spread=0)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = CSVDataset(annotation_path=args.train_file,
                               images_dir=args.images_dir, attributes=attributes,
                               transform=train_transform)
    val_dataset = CSVDataset(annotation_path=args.val_file,
                             images_dir=args.images_dir, attributes=attributes,
                             transform=val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                        num_workers=num_workers)
        , 'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)}

    dataset_sizes_ = {
        'train': len(train_dataset.data), 'val': len(val_dataset.data)
    }

    # dataloaders_ = {
    #     'train': train_dataloader, 'val': val_dataloader
    # }
    netname = 'mnasnet'
    model = MultiOutputModel(trained_labels=train_dataset.attr_names,
                             attrbts=attributes, netname=netname).to(device)
    # normal = 1  # 1- usual 2 - experiment
    if netname == 'mobilenetv2.2':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))  # lr-.001 gamma=.3
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)
    elif netname == 'mnasnet':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=.9)  # lr-.001 gamma=.3
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    elif netname == 'wide_resnet50_2':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))  # lr-.001 gamma=.3
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    else:
        exit(200)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

    logdir = os.path.join(args.work_dir, 'log')
    savedir = os.path.join(args.work_dir, 'save')
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)

    # logger = SummaryWriter(logdir)

    n_train_samples = len(train_dataset.data)

    print("Starting training ...")
    best_acc = 0
    best_acc_epoch = 0
    best_acc_loads = 0
    best_acc_last_load = 0

    # out_model = train_model(model, nn.CrossEntropyLoss(), optimizer, exp_lr_scheduler, dataloaders=dataloaders,
    #                         dataset_sizes=dataset_sizes_, num_epochs=50)

    for epoch in range(start_epoch, N_epochs + 2):
        total_loss = 0
        accuracy = {x: 0 for x in train_dataset.attr.fld_names}
        epoch_start_time = datetime.now()
        for batch in train_dataloader:
            optimizer.zero_grad()
            img = batch['img'].to(device)
            target_labels = batch['labels']
            target_labels = {t: batch['labels'][t].to(device) for t in target_labels}
            output = model(img)
            loss_train, losses_train = model.get_loss(output, target_labels)
            total_loss += loss_train.item()
            batches = calculate_metrics(output, target_labels)
            for x in batches:
                accuracy[x] += batches[x]

            loss_train.backward()
            optimizer.step()

        print(f'epoch:{epoch} loss:{total_loss}   { total_loss / n_train_samples:.4f} ' +
              ' '.join([f'{a}: {accuracy[a] / n_train_samples:.4f}'
                        for a in train_dataset.attr_names]) + f' {datetime.now() - epoch_start_time}')
            # logger.add_scalar('train_loss', total_loss / n_train_samples, epoch)

        chk_point = None
        opt_chk_pnt = None

        # if epoch % 1 == 0:
        cur_best_acc = validate(model=model,
                                dataloader=val_dataloader,
                                fld_names=train_dataset.attr_names,
                                logger=None,
                                iteration=epoch,
                                device=device)
        if cur_best_acc > best_acc:
            best_acc = cur_best_acc
            best_acc_epoch = epoch
            best_acc_last_load = epoch
            best_acc_loads = 0
            f = os.path.join(savedir, f'{args.checkpoint_best}.pth')
            torch.save(model.state_dict(), f)
            fp = os.path.join(savedir, 'checkpoint-best_optim.pth')
            torch.save(optimizer.state_dict(), fp)
            print(f'Stored model on best acc @{best_acc:.4f}')
            if args.graph_out == 'true':
                visualize_grid(model=model,
                               dataloader=val_dataloader,
                               attr=train_dataset.attr,
                               device=device,
                               caption=f'{best_acc:.4f}@{best_acc_epoch}')
        if best_acc_last_load + TRAIN_PERIODE <= epoch:
            f = os.path.join(savedir, f'{args.checkpoint_best}.pth')
            best_acc_loads += 1
            print(f'loaded best thr ptr. {best_acc_loads} time(s) back to: {best_acc_epoch} @{best_acc:.4f}')
            best_acc_last_load = epoch
            model.load_state_dict(torch.load(f, map_location=device))
            optimizer.load_state_dict(torch.load(fp, map_location=device))
            optimizer.step()
            exp_lr_scheduler.step()
            if best_acc_loads > COUNT_OF_LR_STEPS:
                break
#