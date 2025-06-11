
'''
Tuple: (kernel size, number of filters, stride, padding).
M: maxpool (kernel size = [2x2], number of filters = s, stride = 2)
List: Tuples in order, number of repeats.
'''
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock():
    def __init__(self, in_channels, out_channels, **kwargs):
        self.in_channels = in_channels # each channel represent a matrix input R, G, B
        self.out_channels = out_channels
        self.conv = Conv2d(in_channels, out_channels, bias=False, **kwargs) #type of convolution we will be doing
        self.batchnorm = BatchNorm2d(out_channels) #implementing a batch norms on our data
        self.leakyrelu = LeakyRelu(0.1) #the non-linear function we will be using

    def forward(self, x):
        '''
        foreward pass of our nn
        :param x: data
        :return: prediction
        '''
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class YoloV1():
    def __init__(self, in_channels = 3, **kwargs):
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = create_conv_layers(self.architecture)