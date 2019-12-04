import torch
import torch.nn as nn
from torchvision.models import vgg16

class FCN8s(nn.Module):

    def __init__(self, num_of_classes, initialize="xavier"):
        """
            initialize can be "xavier" or "imagenet"
        """
        super(FCN8s, self).__init__()

        self.layers12 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=100),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )
        self.layer3_score = nn.Conv2d(256, num_of_classes, 1)

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )
        self.layer4_score = nn.Conv2d(512, num_of_classes, 1)

        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.layers67 = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.layer8_score = nn.Conv2d(4096, num_of_classes, 1)

        self.layer8_upsample2x = nn.ConvTranspose2d( \
            num_of_classes, num_of_classes, kernel_size=4, stride=2, bias=False)
        self.layer4_upsample2x = nn.ConvTranspose2d( \
            num_of_classes, num_of_classes, kernel_size=4, stride=2, bias=False)
        self.layer3_upsample8x = nn.ConvTranspose2d( \
            num_of_classes, num_of_classes, kernel_size=16, stride=8, bias=False)

        if initialize == "xavier":
            self._initialize_xavier()
        elif initialize == "imagenet":
            self._initialize_imagenet()
        else:
            raise ValueError( \
            "Initialize supports only 'xavier', 'imagenet', and 'zero'. " \
            "Received='{}'".format(initialize))


    def forward(self, input):
        output = self.layers12(input)
        output = self.layer3(output)
        layer3_output = output

        output = self.layer4(output)
        layer4_output = output

        output = self.layer5(output)
        output = self.layers67(output)

        output = self.layer8_score(output)
        layer8_score = output
        layer8_upsample_score = self.layer8_upsample2x(layer8_score)

        output = self.layer4_score(layer4_output)
        layer4_score = output
        # magic
        layer4_score = layer4_score[:, :,
            5:5 + layer8_upsample_score.size()[2],
            5:5 + layer8_upsample_score.size()[3]]

        layer4_upsample_score = self.layer4_upsample2x(
            layer4_score + layer8_upsample_score)

        output = self.layer3_score(layer3_output)
        layer3_score = output
        # magic
        layer3_score = layer3_score[:, :,
            9:9 + layer4_upsample_score.size()[2],
            9:9 + layer4_upsample_score.size()[3]]

        layer3_upsample_score = self.layer3_upsample8x(
            layer3_score + layer4_upsample_score)
        # magic
        layer3_upsample_score = layer3_upsample_score[:, :,
            31:31 + input.size()[2],
            31:31 + input.size()[3]].contiguous()

        return layer3_upsample_score

    def _xavier_module_init(mod):
        if isinstance(mod, nn.Conv2d):
            nn.init.xavier_uniform_(mod.weight)
            mod.bias.data.fill_(.01)
        elif isinstance(mod, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(mod.weight)

    def _initialize_xavier(self):
        self.apply(FCN8s._xavier_module_init)

    def _initialize_imagenet(self):
        pretrained_vgg16 = vgg16(pretrained=True)

        our_features = [
            *list(self.layers12.children()),
            *list(self.layer3.children()),
            *list(self.layer4.children()),
            *list(self.layer5.children())
        ]

        for our_mod, img_mod in zip(our_features, pretrained_vgg16.features):
            if isinstance(img_mod, nn.Conv2d):
                our_mod.weight.data.copy_(img_mod.weight.data)
                our_mod.bias.data.copy_(img_mod.bias.data)


        our_mods = (self.layers67[0], self.layers67[3])
        idx = 0
        for img_mod in pretrained_vgg16.classifier:
            if isinstance(img_mod, nn.Linear):
                m = our_mods[idx]
                m.weight.data.copy_(img_mod.weight.data.view(m.weight.size()))
                m.bias.data.copy_(img_mod.bias.data.view(m.bias.size()))
                idx = idx + 1

            if idx > 1:
                break
