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
        # if use_resnet:
        #     self.base_model = models.resnet.resnet50(pretrained=True)  # take the model without classifier
        #     lt_chan = self.base_model.fc.shape[0]  # size of the layer before classifier
        # else:
        print(f'[MODEL] mobilenet_v2 loaded')
        self.base_model = models.mobilenet_v2(pretrained=True).features  # take the model without classifier
        lt_chan = models.mobilenet_v2().last_channel  # size of the layer before classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.outs = {}
        for lbl in trained_labels:
            count = attrbts.num_labels[lbl]
            v = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=lt_chan, out_features=count)
            )
            # self.outs[lbl] = v
            self.add_module(lbl, v)

        # for ll in list(self._modules):
        #     print('\n', ll)

    def forward(self, x):
        x = self.base_model(x)
        # print('x', x.device)
        x = self.pool(x)
        # print('x', x.device)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)
        # print('x', x.device)
        #     print(f'\n\n{lbl}:{count}\n', self.__dict__[lbl])
        # # create separate classifiers for our outputs
        # self.color = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=lt_chan, out_features=n_color_classes)
        # )
        # self.gender = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=lt_chan, out_features=count)
        # )
        # self.article = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=lt_chan, out_features=n_article_classes)
        # )

        return {an: self._modules[an](x) for an in self.fld_names}
        #     'color': self.color(x),
        #     'gender': self.gender(x),
        #     'article': self.article(x)
        # }

    def get_loss(self, net_output, ground_truth):
        out = {an: F.cross_entropy(net_output[an], ground_truth[an]) for an in self.fld_names}
        loss = sum([xx for x, xx in out.items()]) / len(self.fld_names)
        return loss, out
