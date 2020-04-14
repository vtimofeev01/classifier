import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from dl_src.dataset import AttributesDataset


class MultiOutputModel(nn.Module):

    def __init__(self, trained_labels: list, attrbts: AttributesDataset):
        super().__init__()
        self.fld_names = trained_labels
        print(f'Model trained attributes: {self.fld_names}')
        self.base_model = models.mobilenet_v2().features  # take the model without classifier
        lt_chan = models.mobilenet_v2().last_channel  # size of the layer before classifier

        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # self.a0 = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=lt_chan,
        #                                                      out_features=attrbts.num_labels[trained_labels[0]]))
        # for lbl, count in attributes.num_labels.items():
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
