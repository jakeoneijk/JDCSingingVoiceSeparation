import torch
from torch import nn
from .ResNet import ResNet
from .InitializeMethod import InitializeMethod

class JDCNet(nn.Module):
    def __init__(self, num_class = 722, seq_len = 31, leaky_relu_slope=0.01):
        super(JDCNet,self).__init__()
        self.seq_length = seq_len
        self.num_class = num_class

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1,
            bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(64,64, 3, padding=1, bias=False)
        )
        
        self.res_block1 = ResNet(input_channel=64, output_channel=128)
        self.res_block2 = ResNet(input_channel=128, output_channel=192)
        self.res_block3 = ResNet(input_channel=192, output_channel=256)

        self.pool_block = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1,4)),
            nn.Dropout(p=0.5),
        )

        self.maxpool_1 = nn.MaxPool2d(kernel_size=(1, 256))
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(1, 64))
        self.maxpool_3 = nn.MaxPool2d(kernel_size=(1, 16))

        self.detector_conv = nn.Sequential(
            nn.Conv2d(640, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Dropout(p=0.5),
        )

        self.bilstm_classifier = nn.LSTM(
            input_size=512, hidden_size=256,
            batch_first=True, dropout=0.3, bidirectional=True)

        self.bilstm_detector = nn.LSTM(
            input_size=512, hidden_size=256,
            batch_first=True, dropout=0.3, bidirectional=True)
        
        self.classifier = nn.Linear(in_features=512, out_features=self.num_class)
        self.detector = nn.Linear(in_features=512, out_features=2)

        self.apply(InitializeMethod().init_weights)
    
    def forward(self,x):
        ###############################
        # forward pass for classifier #
        ###############################
        convblock_out = self.conv_block(x)

        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)
        poolblock_out = self.pool_block(resblock3_out)

        classifier_out = poolblock_out.permute(0, 2, 1, 3).contiguous().view((-1, 31, 512))
        classifier_out, _ = self.bilstm_classifier(classifier_out)

        classifier_out = classifier_out.contiguous().view((-1, 512))  # (b * 31, 512)
        classifier_out = self.classifier(classifier_out)
        classifier_out = classifier_out.view((-1, 31, self.num_class))

        mp1_out = self.maxpool_1(convblock_out)
        mp2_out = self.maxpool_2(resblock1_out)
        mp3_out = self.maxpool_3(resblock2_out)

        concat_out = torch.cat((mp1_out, mp2_out, mp3_out, poolblock_out), dim=1)
        detector_out = self.detector_conv(concat_out)

        detector_out = detector_out.permute(0, 2, 1, 3).contiguous().view((-1, 31, 512))
        detector_out, _ = self.bilstm_detector(detector_out)

        detector_out = detector_out.contiguous().view((-1, 512))
        detector_out = self.detector(detector_out)
        detector_out = detector_out.view((-1, 31, 2))

        pitch_pred, nonvoice_pred = torch.split(classifier_out, [self.num_class - 1, 1], dim=2)

        classifier_detection = torch.cat(
            (torch.sum(pitch_pred, dim=2, keepdim=True), nonvoice_pred), dim=2)

        detector_out = detector_out + classifier_detection
        return classifier_out, detector_out

if __name__ == '__main__':
    print("model size test")
    dummy = torch.randn((10, 1, 31, 513))  # dummy random input
    jdc = JDCNet()
    clss, detect = jdc(dummy)
    print(clss.size())
    print(detect.size())
        


