from dataset import *

VOC_set = DatasetLoader("VOC2012/JPEGImages", "VOC2012/Annotations")

x = VOC_set.__getitem__(1)

print(x)