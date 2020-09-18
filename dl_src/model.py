import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import init
from dl_src.dataset import AttributesDataset

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class MultiOutputModel(nn.Module):

    def __init__(self, trained_labels: list, attrbts: AttributesDataset, netname='mobilenetv2.2',
                 freeze=False, first_stage_dict = None):
        self.netname = netname
        self.first_stage_dict = first_stage_dict
        if netname == 'mobilenetv2':
            super().__init__()
            self.fld_names = trained_labels
            print(f'Model trained attributes: {self.fld_names}')
            self.base_model = models.mobilenet_v2(pretrained=True).features  # take the model without classifier
            lt_chan = models.mobilenet_v2().last_channel  # size of the layer before classifier
            print(f'backbone output shape: {lt_chan}')
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            print(f'pool={self.pool}')
            if freeze:
                for param in self.base_model.parameters():
                    param.requires_grad = False
            self.outs = {}
            for lbl in trained_labels:
                count = attrbts.num_labels[lbl]
                print(f'vreated <<{lbl}>> in_features={lt_chan}, out_features={count} ')
                v = nn.Sequential(
                    nn.Dropout(p=0.2, inplace=False),
                    nn.Linear(in_features=lt_chan, out_features=count)
                )
                # print(v)
                self.add_module(lbl, v)
            for a in self._modules:
                print('modules:', a)
                
        elif netname == 'mobilenetv2.2':
            super().__init__()
            self.fld_names = trained_labels
            print(f'mobilenetv2.2 trained attributes: {self.fld_names}')
            self.base_model = models.mobilenet_v2(pretrained=True)  #
            in_ftrs = self.base_model.last_channel
            # self.pool = nn.AdaptiveAvgPool2d((1, 1))
            for lbl in trained_labels:
                print(trained_labels, lbl)
                count = attrbts.num_labels[lbl]
                print(f'created <<{lbl}>> in_features={in_ftrs}, out_features={count} ')
                v = nn.Sequential(
                    nn.Dropout(p=0.2, inplace=False),
                    nn.Linear(in_features=in_ftrs, out_features=count)
                )
                self.base_model.classifier = v
                self.lll = lbl
            for a in self._modules:
                print('modules:', a)
        elif netname == 'mnasnet':
            super().__init__()
            self.fld_names = trained_labels
            print(f'mnasnet Model trained attributes: {self.fld_names}')
            self.base_model = models.mnasnet1_0(pretrained=True).layers  # take the model without classifier
            lt_chan = 1280  # self.base_model.last_channel  # size of the layer before classifier
            print(f'backbone output shape: {lt_chan}')
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.outs = {}
            for lbl in trained_labels:
                count = attrbts.num_labels[lbl]
                print(f'created <<{lbl}>> in_features={lt_chan}, out_features={count} ')
                v = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=lt_chan, out_features=count)
                )
                self.add_module(lbl, v)
            for a in self._modules:
                print('modules:', a)
        elif netname == 'mnasnet-2':
            super().__init__()
            self.fld_names = trained_labels
            print(f'mnasnet-2 Model trained attributes: {self.fld_names}')
            self.base_model = models.mnasnet1_0(pretrained=True).layers  # take the model without classifier
            lt_chan = 1280  # self.base_model.last_channel  # size of the layer before classifier
            print(f'backbone output shape: {lt_chan}')
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.outs = {}
            for lbl in trained_labels:
                count = attrbts.num_labels[lbl]
                print(f'created <<{lbl}>> in_features={lt_chan}, out_features={count} ')
                v = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=lt_chan, out_features=count)
                )
                self.add_module(lbl, v)
            # for a in self._modules:
        elif netname == 'wide_resnet50_2':
            super().__init__()
            self.fld_names = trained_labels
            print(f'wide_resnet50_2 Model trained attributes: {self.fld_names}')
            self.base_model = models.wide_resnet50_2(pretrained=True)  #
            in_ftrs = self.base_model.fc.in_features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            for lbl in trained_labels:
                print(trained_labels, lbl)
                count = attrbts.num_labels[lbl]
                print(f'created <<{lbl}>> in_features={in_ftrs}, out_features={count} ')
                v = nn.Sequential(
                    # nn.Dropout(p=0.2),
                    nn.Linear(in_features=in_ftrs, out_features=count)
                )
                self.base_model.fc = v
                self.lll = lbl
            for a in self._modules:
                print('modules:', a)


    def forward(self, x):
        if self.netname == 'mobilenetv2':
            x = self.base_model(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return {an: self._modules[an](x) for an in self.fld_names}
        elif self.netname == 'mnasnet':
            x = self.base_model(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return {an: self._modules[an](x) for an in self.fld_names}
        elif self.netname == 'wide_resnet50_2':
            return {self.lll: self.base_model(x)}
        elif self.netname == 'mobilenetv2.2':
            return {self.lll: self.base_model(x)}

    def get_loss(self, net_output, ground_truth):
        out = {an: F.cross_entropy(net_output[an], ground_truth[an]) for an in self.fld_names}
        loss = sum([xx for x, xx in out.items()]) / len(self.fld_names)
        return loss, out


class Backbone_nFC(nn.Module):
    def __init__(self, class_num, model_name='resnet50_nfc'):
        super(Backbone_nFC, self).__init__()
        self.model_name = model_name
        self.backbone_name = model_name.split('_')[0]
        self.class_num = class_num

        model_ft = getattr(models, self.backbone_name)(pretrained=True)
        if 'resnet' in self.backbone_name:
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft
            self.num_ftrs = 2048
        elif 'densenet' in self.backbone_name:
            model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft.features
            self.num_ftrs = 1024
        else:
            raise NotImplementedError

        for c in range(self.class_num):
            self.__setattr__('class_%d' % c, ClassBlock(input_dim=self.num_ftrs, class_num=1, activ='sigmoid'))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        pred_label = [self.__getattr__('class_%d' % c)(x) for c in range(self.class_num)]
        pred_label = torch.cat(pred_label, dim=1)
        return pred_label


class Backbone_nFC_Id(nn.Module):
    def __init__(self, class_num, id_num, model_name='resnet50_nfc_id'):
        super(Backbone_nFC_Id, self).__init__()
        self.model_name = model_name
        self.backbone_name = model_name.split('_')[0]
        self.class_num = class_num
        self.id_num = id_num

        model_ft = getattr(models, self.backbone_name)(pretrained=True)
        if 'resnet' in self.backbone_name:
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft
            self.num_ftrs = 2048
        elif 'densenet' in self.backbone_name:
            model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft.features
            self.num_ftrs = 1024
        else:
            raise NotImplementedError

        for c in range(self.class_num + 1):
            if c == self.class_num:
                self.__setattr__('class_%d' % c, ClassBlock(self.num_ftrs, class_num=self.id_num, activ='none'))
            else:
                self.__setattr__('class_%d' % c, ClassBlock(self.num_ftrs, class_num=1, activ='sigmoid'))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        pred_label = [self.__getattr__('class_%d' % c)(x) for c in range(self.class_num)]
        pred_label = torch.cat(pred_label, dim=1)
        pred_id = self.__getattr__('class_%d' % self.class_num)(x)
        return pred_label, pred_id


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num=1, activ='sigmoid', num_bottleneck=512):
        super(ClassBlock, self).__init__()

        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        if activ == 'sigmoid':
            classifier += [nn.Sigmoid()]
        elif activ == 'softmax':
            classifier += [nn.Softmax()]
        elif activ == 'none':
            classifier += []
        else:
            raise AssertionError("Unsupported activation: {}".format(activ))
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
