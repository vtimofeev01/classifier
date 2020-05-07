import argparse
import csv
import os
import warnings
from datetime import datetime
from PIL import Image
import torch
import torchvision.transforms as transforms
from dl_src.dataset import CSVDataset, AttributesDataset, mean, std
from dl_src.model import MultiOutputModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
from torch import onnx


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    try:
        epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    except:
        epoch = 20
    return epoch


def validate(model, dataloader, fld_names, logger, iteration, device, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        total_loss = 0
        accuracy = {x: 0 for x in fld_names}
        epoch_start_time = datetime.now()
        for batch in dataloader:
            img = batch['img'].to(device)
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img)

            val_train, val_train_losses = model.get_loss(output, target_labels)
            total_loss += val_train.item()
            batches = calculate_metrics(output, target_labels)
            for x in batches:
                accuracy[x] += batches[x]
        totalacc = sum([y for x, y in accuracy.items()])

    n_samples = len(dataloader)
    print(f'epoch:{iteration} val loss: {total_loss / n_samples:.4f} ' +
          ' '.join([f'{a}: {accuracy[a] / n_samples:.4f}'
                    for a in fld_names]) + f' {datetime.now() - epoch_start_time}'
                                           f' total_acc:{totalacc / n_samples:.4f}')
    model.train()
    return totalacc /n_samples


def visualize_grid(model, dataloader, attr, device, show_cn_matrices=True, checkpoint=None, csv_filename=None):
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

    if list_of_images_to_check:
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


def calculate_metrics(output, target):
    predicts = {x: output[x].cpu().max(1)[1] for x in output}
    gts = {x: target[x].cpu() for x in target}

    with warnings.catch_warnings():
        # sklearn may produce a warning when processing zero row in confusion matrix
        warnings.simplefilter("ignore")
        accuracy = {x: balanced_accuracy_score(y_true=gts[x].numpy(),
                                               y_pred=predicts[x].numpy()) for x in output}

    return accuracy


def cut_pil_image(image: Image, border=20):
    w, h = image.size
    return image.crop((border, border, w - border, h - border))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the checkpoint")
    parser.add_argument('--images_dir', type=str, required=True,
                        help="Folder containing images described in CSV file")
    parser.add_argument('--test_file', type=str, required=True,
                        help="CSV-file format (image name, label1, label2, ...) to use for training")
    parser.add_argument('--attributes_file', type=str,
                        help="Path to the file with attributes")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    if args.attributes_file is None:
        args.attributes_file = args.test_file

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    attributes = AttributesDataset(args.attributes_file)

    # during validation we use only tensor and normalization transforms
    val_transform = transforms.Compose([
        transforms.Lambda(lambd=cut_pil_image),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_dataset = CSVDataset(annotation_path=args.test_file,
                              images_dir=args.images_dir,
                              attributes=attributes,
                              transform=val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    model = MultiOutputModel(trained_labels=test_dataset.attr_names,
                             attrbts=attributes).to(device)

    # Visualization of the trained model
    csv_filename = os.path.join(
        os.path.split(args.test_file)[0], 'files_to_check.csv'
    )
    visualize_grid(model=model,
                   dataloader=test_dataloader,
                   attr=attributes,
                   device=device, checkpoint=args.checkpoint,
                   csv_filename=csv_filename)

    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    core_name = f'mnv2_single_attribute'
    input_names = ["input1"]  # + [ "learned_%d" % i for i in range(16) ]
    output_names = ["output1"]
    for data_type in ['FP16', 'FP32']:
        onnx.export(model=model, args=dummy_input,
                    f=f"{core_name}.onnx",
                    verbose=True,
                    input_names=input_names,
                    output_names=output_names)

        # mo_run = f'python3 /opt/intel/openvino/deployment_tools/model_optimizer/' \
        #          f'mo.py --input_model {core_name}.onnx --data_type FP16' \
        #          f'--output_dir {os.curdir}'
        mo_run = f'python3 /opt/intel/openvino/deployment_tools/model_optimizer/' \
                 f'mo.py --input_model {core_name}.onnx  ' \
                 f"--reverse_input_channels " \
                 f"--mean_values=[123.675,116.28,103.53] " \
                 f'--scale_values=[58.395,57.12,57.375]  ' \
                 f'--data_type {data_type}' \
                 f' --output_dir {os.curdir}/{data_type}'
        print('\n', mo_run, '\n')
        os.system(mo_run)

        json_object = json.dumps({'labels_id_to_name': attributes.labels_id_to_name,
                                  'labels_name_to_id': attributes.labels_name_to_id}, indent=4)
        with open(f'{os.curdir}/{data_type}/{core_name}.json', "w") as outfile:
            outfile.write(json_object)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
