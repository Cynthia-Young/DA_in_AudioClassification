import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
            x = self.dropout(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=9, type="linear"):
        super(feat_classifier, self).__init__()
        if type == "linear":
            self.fc = nn.Linear(bottleneck_dim, class_num)
        else:
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        encoder = nn.Sequential()
        encoder.add_module("conv1", nn.Conv2d(2, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
        encoder.add_module("relu1", nn.ReLU())
        encoder.add_module("bn1", nn.BatchNorm2d(8))
        encoder.add_module("conv2", nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
        encoder.add_module("relu2", nn.ReLU())
        encoder.add_module("bn2", nn.BatchNorm2d(16))
        encoder.add_module("conv3", nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
        encoder.add_module("relu3", nn.ReLU())
        encoder.add_module("bn3", nn.BatchNorm2d(32))
        self.encoder = encoder

         # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(0.5)
        self.lin = nn.Linear(in_features=32, out_features=9)

    def forward(self, inp_x):
        inp_x = self.encoder(inp_x)
        inp_x = self.ap(inp_x)
        inp_x = inp_x.view(inp_x.shape[0], -1)
        inp_x = self.dropout(inp_x)
        inp_x = self.lin(inp_x)
        return inp_x