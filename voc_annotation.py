import xml.etree.ElementTree as ET
from os import getcwd

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

wd = getcwd()
classes = ["Yellow", "Green", "Blue", "Cyan", "Orange", "Red", "Seat", "Other"]
#classes = ["Bike", "Seat"]

def convert_annotation(year, image_id, list_file):
    in_file = open('F:\project\efficientdet-pytorch-Bike\VOCdevkit\My_Dataset\Annotations\%s.xml'%(image_id))
              #open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    if root.find('object')==None:
        return
    list_file.write('../efficientdet-pytorch-Bike\VOCdevkit\My_Dataset/JPEGImages/%s.jpg'%(image_id))

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:        # or int(difficult)==1:
            cls = "Yellow"  #防止undefined  cls = "Bike"  #
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    list_file.write('\n')

for year, image_set in sets:
    image_ids = open('./VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')

    for image_id in image_ids:
        convert_annotation(year, image_id, list_file)
    list_file.close()
