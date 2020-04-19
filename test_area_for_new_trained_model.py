#!/usr/bin/env python
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from time import time

import torch
from openvino.inference_engine import IENetwork, IECore
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from dl_src.dataset import make_list_of_files, AttributesDataset, CSVDataset
from dl_src.model import MultiOutputModel
from train import cut_pil_image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    args.add_argument("-a", "--attributes_file", help="attributes file.", required=True,
                      type=str)
    args.add_argument("-checkpoint", "--checkpoint", help="checkpoint for PyTorch", required=True,
                      type=str)
    args.add_argument("-images_dir", "--images_dir", help="images_dir", required=True,
                      type=str)
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                      required=True,
                      type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                           " kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to a labels mapping file", default=None, type=str)
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)

    return parser


def main():
    # log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    trfm = transforms.Compose([
        transforms.Lambda(lambd=cut_pil_image),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # Plugin initialization for specified device and load extensions library if specified
    print("Creating Inference Engine")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    # Read IR
    print("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            print("Following layers are not supported by the plugin for specified device {}:\n {}".
                  format(args.device, ', '.join(not_supported_layers)))
            print("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                  "or --cpu_extension command line argument")
            sys.exit(1)

    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    print("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    # Read and pre-process input images
    n, c, h, w = net.inputs[input_blob].shape
    print(f'input {input_blob}: {net.inputs[input_blob].shape}')
    print(f'input {out_blob}: {net.outputs[out_blob].shape}')
    bs, ots = net.outputs[out_blob].shape
    print(f'read from:{args.input}')

    # print(f'labels:{labels}')
    # images = [os.path.join(a, b) for a, b in zip(*make_list_of_files(args.input))]

    print("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)


    attributes = AttributesDataset(args.attributes_file)
    test_dataset = CSVDataset(annotation_path=args.attributes_file,
                              images_dir=args.images_dir,
                              attributes=attributes,
                              transform=trfm)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    model = MultiOutputModel(trained_labels=attributes.fld_names,
                             attrbts=attributes).to(device)
    statedict = torch.load(args.checkpoint, map_location='cuda')
    model.load_state_dict(statedict)
    model.eval()
    with torch.no_grad():
        for image in test_dataloader:
            img = cv2.imread(image['img_path'][0])
            print(f'image shape={img.shape}: {image["img_path"][0]}')
            cv2.imshow('xxxx', img)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            img = img.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            res = exec_net.infer(inputs={input_blob: img})[out_blob]
            res2 = model(image['img'].to(device))
            for il, (v, v2) in enumerate(zip(res[0], res2['label'][0])):
                l = attributes.labels_id_to_name["label"][il]
                print(f'{il} {l:.<18} {v:+.4f} ... {v2:+.4f}')
            k = cv2.waitKey(0)
            if k == ord('q'):
                exit()


if __name__ == '__main__':
    sys.exit(main() or 0)
