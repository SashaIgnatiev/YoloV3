import os
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET


class DatasetLoader:
    def __init__(self, image_dir, annotations_dir, transform=None):
        self.annotations_dir = annotations_dir #directory for the image descriptions
        self.image_dir = image_dir #directory for the image file
        self.transform = transform #transformation applied to the image (not used as of now, but set up for future implementation)

    def __len__(self):
        '''
        Getter method for the length of the dataset.
        :return: # items in dataset
        '''
        return len(os.listdir(self.annotations_dir))

    def __getitem__(self, idx):
        '''
        Method for getting an item from the dataset for training/testing.
        :param idx:
        :return: tuple containing the image matrix, and the 'true' positions of each object
        '''

        #obtaining the image name
        entries = os.listdir(self.annotations_dir)
        sorted_entries = sorted(entries)
        name = sorted_entries[idx].split('.')[0]

        #grabbing the image matrix
        img_path = os.path.join(self.image_dir, name + ".jpg")
        img = Image.open(img_path)
        img_array = np.array(img)

        #grabbing the annotations
        annotaion_path = os.path.join(self.annotations_dir, name + ".xml")
        tree = ET.parse(annotaion_path)
        root = tree.getroot()

        #obtaining the image size for verification
        dim_list = list()
        for type_tag in root.findall('size'):
            for child in type_tag:
                dim_list.append(int(child.text))
        #switching the order around, so it is in the same order as the dimensions of the array
        x = dim_list[0]
        dim_list[0] = dim_list[1]
        dim_list[1] = x
        dims = tuple(dim_list)


        #Getting the object names

        data = list()
        for object in root.iter('object'):
            name = object.find('name').text
            xmin = object.find('bndbox/xmin').text
            ymin = object.find('bndbox/ymin').text
            xmax = object.find('bndbox/xmax').text
            ymax = object.find('bndbox/ymax').text
            data_tuple = (name, int(xmin), int(ymin), int(xmax), int(ymax))
            data.append(data_tuple)


        try:
            assert dims == img_array.shape
            return (data, img_array)
        except:
            print(dims)
            print(img_array.shape)
            print("you might be working with 2 different images, something is wrong :(")
            return None
