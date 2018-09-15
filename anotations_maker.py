import xml.etree.ElementTree as ET
import pascal_voc_writer as pvw

default_dir='/home/output/'

def label_image(image_path, h, w, name, xmin, ymin, xmax, ymax, main_dir = default_dir):
    base_name = image_path.split('/')[-1]
    out_xml = main_dir+'{}.xml'.format(base_name.split('.')[0])
    writer = pvw.Writer(image_path, h, w)
    writer.addObject(name, xmin, ymin, xmax, ymax)
    writer.save(out_xml)
    correct_path(out_xml, image_path)

def correct_path(in_xml, image_path):
    tree = ET.parse(in_xml)
    root = tree.getroot()
    folder =image_path.split('/')[-2]
    print('folder: '+ folder )
    for elem in root.iter('path'):
        elem.text = image_path
    for elem in root.iter('folder'):
        elem.text = folder
    tree.write(in_xml)
