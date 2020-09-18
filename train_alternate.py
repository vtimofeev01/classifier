# !/usr/local/bin/python3
import os
import time
import argparse
import matplotlib
import torchvision.transforms as transforms
from dl_src.dataset import CSVDataset, AttributesDataset, CSVDataset2
from train import cut_pil_image
from torch.utils.data import DataLoader

matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from net import get_model

######################################################################
# Settings
# --------
use_gpu = True
dataset_dict = {
    'market': 'Market-1501',
    'duke': 'DukeMTMC-reID',
}

######################################################################
# Argument
# --------
parser = argparse.ArgumentParser(description='Training')
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
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--num_workers', default=2, type=int, help='num_workers')
parser.add_argument('--device', type=str, default='cuda', help="Device: 'cuda' or 'cpu'")
parser.add_argument('--checkpoint_best', type=str, default='checkpoint-best', help="checkpoint-best")
parser.add_argument('--backbone', default='resnet50', type=str,
                    help='backbone: resnet50, resnet34, resnet18, densenet121')
parser.add_argument('--lamba', default=1.0, type=float, help='weight of id loss')
args = parser.parse_args()

# assert args.dataset in ['market', 'duke']
assert args.backbone in ['resnet50', 'resnet34', 'resnet18', 'densenet121']

# dataset_name = dataset_dict[args.dataset]
# model_name = '{}_nfc_id'.format(args.backbone) if args.use_id else '{}_nfc'.format(args.backbone)
model_name = '{}_nfc'.format(args.backbone)
# data_dir = args.data_path
model_dir = os.path.join('./checkpoints', args.checkpoint_best, model_name)

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)


######################################################################
# Function
# --------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(model_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if use_gpu:
        network.cuda()
    print('Save model to {}'.format(save_path))


######################################################################
# Draw Curve
# -----------
x_epoch = []
y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join(model_dir, 'train.jpg'))


######################################################################
# DataLoader
# ---------
image_datasets = {}

if args.attributes_file is None:
    args.attributes_file = args.train_file
mean = [0.485, 0.456, 0.406]  # [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]  # [0.229, 0.224, 0.225]
attributes = AttributesDataset(os.path.join(args.work_dir, 'data.csv'))
# specify image transforms for augmentation during training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
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

image_datasets['val'] = CSVDataset2(annotation_path=args.val_file,
                                   images_dir=args.images_dir, attributes=attributes,
                                   transform=val_transform)

image_datasets['train'] = CSVDataset2(annotation_path=args.train_file,
                                     images_dir=args.images_dir, attributes=attributes,
                                     transform=train_transform)

dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers)  # , drop_last=True)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# images, indices, labels, ids, cams, names = next(iter(dataloaders['train']))

num_label = len(attributes.labels)  # image_datasets['train'].num_label()
# num_id = image_datasets['train'].num_id()
# labels_list = image_datasets['train'].labels()


######################################################################
# Model and Optimizer
# ------------------
model = get_model(model_name, num_label)
if use_gpu:
    model = model.cuda()

# loss
criterion_bce = nn.BCELoss()
criterion_ce = nn.CrossEntropyLoss()

# optimizer
ignored_params = list(map(id, model.features.parameters()))
classifier_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer = torch.optim.SGD([
    {'params': model.features.parameters(), 'lr': 0.01},
    {'params': classifier_params, 'lr': 0.1},
], momentum=0.9, weight_decay=5e-4, nesterov=True)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)


######################################################################
# Training the model
# ------------------
def train_model(model, optimizer, scheduler, num_epochs):
    since = time.time()

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for count, (images, labels, indices) in enumerate(dataloaders[phase]):
                # get the inputs
                labels = labels.float()
                if use_gpu:
                    images = images.cuda()
                    labels = labels.cuda()
                    # indices = indices.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward

                pred_label = model(images)

                total_loss = criterion_bce(pred_label, labels)
                # backward + optimize only if in training phase
                if phase == 'train':
                    total_loss.backward()
                    optimizer.step()

                preds = torch.gt(pred_label, torch.ones_like(pred_label) / 2)
                # statistics
                running_loss += total_loss.item()
                running_corrects += torch.sum(preds == labels.byte()).item() / num_label
                if count % 1000 == 0:
                    print('step: ({}/{})  |  label loss: {:.4f}'.format(
                        count * args.batch_size, dataset_sizes[phase], total_loss.item()))


            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch % 5 == 0:
                    save_network(model, epoch)
                draw_curve(epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')


######################################################################
# Main
# -----
train_model(model, optimizer, exp_lr_scheduler, num_epochs=args.n_epochs)
