import argparse
import os
from datetime import datetime
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from dl_src.dataset import CSVDataset, AttributesDataset, mean, std
from dl_src.model import MultiOutputModel
from test import calculate_metrics, validate
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import csv


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch


def visualize_grid(model, dataloader, attr, device, show_cn_matrices=True, checkpoint=None, csv_filename=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)
    model.eval()

    # imgs = []
    gt_labels = {x: list() for x in attr.fld_names}
    gt_all = {x: list() for x in attr.fld_names}
    predicted_all = {x: list() for x in attr.fld_names}
    accuracyes = {x: 0 for x in attr.fld_names}

    list_of_images_to_check = []
    loitc_columns = ['filename', 'choice']


    # gt_labels = []
    # gt_color_all = []
    # gt_gender_all = []
    # gt_article_all = []
    # predicted_color_all = []
    # predicted_gender_all = []
    # predicted_article_all = []

    # accuracy_color = 0
    # accuracy_gender = 0
    # accuracy_article = 0

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
            ConfusionMatrixDisplay(cn_matrix, attr.labels[x]).plot(
                include_values=True, xticks_rotation='vertical')
            plt.title(x)
            plt.tight_layout()
            plt.show()

    model.train()


def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def checkpoint_save(model, name, epoch):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)
    return f


def cut_pil_image(image: Image, border=20):
    w, h = image.size
    return image.crop((border, border, w-border, h-border))


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
    parser.add_argument('--train_val', type=float, default=.7,
                        help="Part of the dataset that will be used for training. Rest - for validation")
    parser.add_argument('--n_epochs', type=int, default=50, help="number of training epoch's")
    parser.add_argument('--batch_size', type=int, default=32, help="number of training epoch's")
    parser.add_argument('--num_workers', type=int, default=10, help="number of workers")
    parser.add_argument('--device', type=str, default='cuda', help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    start_epoch = 1
    N_epochs = args.n_epochs
    batch_size = args.batch_size
    num_workers = args.num_workers  # number of processes to handle dataset loading
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    if args.attributes_file is None:
        args.attributes_file = args.train_file

    attributes = AttributesDataset(args.attributes_file)

    # specify image transforms for augmentation during training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), # scale=(0.8, 1),
                                shear=None, resample=False, fillcolor=(150, 150, 150)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Lambda(lambd=cut_pil_image),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # during validation we use only tensor and normalization transforms
    val_transform = transforms.Compose([
        transforms.Lambda(lambd=cut_pil_image),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    whole_dataset = CSVDataset(annotation_path=args.train_file, images_dir=args.images_dir, attributes=attributes,
                               transform=train_transform)
    ll = len(whole_dataset)
    train_len = int(ll * args.train_val)
    valid_len = ll - train_len
    train_dataset, val_dataset = random_split(whole_dataset, [train_len, valid_len])
    val_dataset.transform = val_transform

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = MultiOutputModel(trained_labels=whole_dataset.attr_names,
                             attrbts=attributes).to(device)

    optimizer = torch.optim.Adam(model.parameters())

    logdir = os.path.join(args.work_dir)
    savedir = os.path.join(args.work_dir)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    logger = SummaryWriter(logdir)

    n_train_samples = len(train_dataloader)

    print("Starting training ...")

    for epoch in range(start_epoch, N_epochs + 1):
        total_loss = 0
        accuracy = {x: 0 for x in whole_dataset.attr.fld_names}
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

        print(f'epoch:{epoch} ' +
              ' '.join([f'{a}: {accuracy[a] / n_train_samples:.4f}'
                        for a in whole_dataset.attr_names]) + f' {datetime.now() - epoch_start_time}')
        logger.add_scalar('train_loss', total_loss / n_train_samples, epoch)

        chk_point = None

        if epoch % 5 == 0:
            validate(model=model,
                     dataloader=val_dataloader,
                     fld_names=whole_dataset.attr_names,
                     logger=logger,
                     iteration=epoch,
                     device=device)

        if epoch % 25 == 0:
            chk_point = checkpoint_save(model, savedir, epoch)
            visualize_grid(model=model,
                           dataloader=val_dataloader,
                           attr=whole_dataset.attr,
                           device=device)