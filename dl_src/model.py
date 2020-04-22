import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import resnet50

from dl_src.dataset import AttributesDataset

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class MultiOutputModel(nn.Module):

    def __init__(self, trained_labels: list, attrbts: AttributesDataset, use_resnet=False):
        super().__init__()
        self.fld_names = trained_labels
        print(f'Model trained attributes: {self.fld_names}')
        self.base_model = models.mobilenet_v2(pretrained=True).features  # take the model without classifier
        lt_chan = models.mobilenet_v2().last_channel  # size of the layer before classifier
        print(f'backbone output shape: {lt_chan}')
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        print(f'pool={self.pool}')
        self.outs = {}
        for lbl in trained_labels:
            count = attrbts.num_labels[lbl]
            print(f'vreated <<{lbl}>> in_features={lt_chan}, out_features={count} ')
            v = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=lt_chan, out_features=count)
            )
            # print(v)
            self.add_module(lbl, v)
        for a in self._modules:
            print('modules:', a)


    def forward(self, x):
        # print(x.shape)
        x = self.base_model(x)
        # print(x.shape)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        # return {'label': self._modules['label'](x)}
        return {an: self._modules[an](x) for an in self.fld_names}


    def get_loss(self, net_output, ground_truth):
        out = {an: F.cross_entropy(net_output[an], ground_truth[an]) for an in self.fld_names}
        loss = sum([xx for x, xx in out.items()]) / len(self.fld_names)
        return loss, out
