from helper_methods import *
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

# TODO: implement Conv2d, Sequential, Batchnorm2D, flatten, linear

class CNNBlock():

    def __init__(self, in_channels, out_channels, **kwargs):
        self.in_channels = in_channels # each channel represent a matrix input R, G, B
        self.out_channels = out_channels
        self.conv = Conv2d(in_channels, out_channels, **kwargs) #type of convolution we will be doing
        self.batchnorm = BatchNorm2d(out_channels) #implementing a batch norms on our data
        self.leakyReLU = np.vectorize(LeakyReLU(self, 0.1)) #the non-linear function we will be using, applied elementwise
        # to the tensor

    def forward(self, x):
        '''
        forward pass of the nn
        :param x: data
        :return: output of nn
        '''
        return self.leakyReLU(self.batchnorm(self.conv(x)))


class YoloV1():
    def __init__(self, in_channels = 3, **kwargs):
        super(YoloV1, self).__init__() #not sure what this does, ask Chad lol
        self.architecture = architecture_config #imports the architechture d
        self.in_channels = in_channels
        self.darknet = self.create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(flatten(x, start_dim=1))

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += CNNBlock(
                    in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3]
                )
            elif type(x) == str:
                layers += [MaxPool2d(kernel_size=2, stride=2)]

            elif type(x) == list:
                conv1 = x[0] #tuple
                conv2 = x[1] #tuple
                num_repeats = x[2] #integer

                for i in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3]
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3]
                        )
                    ]

                    in_channels = conv2[1]

        return Sequantial(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes,  **kwargs):
        S, B, C = split_size, num_boxes, num_classes
        return Sequential(
            Flatten(),
            Linear(1024 * S * S, 496), #Original paper this should be 4096
            Dropout(0.5),
            leakyReLU(0.1),
            Linear(496, S * S (C + B * 5)), #(S, S, 30) where C + B * 5 = 30
        )

