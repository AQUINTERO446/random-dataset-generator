import xml.etree.ElementTree as ET
import pascal_voc_writer as pvw

default_dir='/home/output/'
writer = ''
out_xml = ''
image_path = ''

def set_writer(in_image_path, h, w, main_dir = default_dir):
    global  image_path, out_xml, writer
    image_path = in_image_path
    writer = pvw.Writer(image_path, h, w)
    base_name = image_path.split('/')[-1]
    out_xml = main_dir+'{}.xml'.format(base_name.split('.')[0])

def label_image(name, xmin, ymin, xmax, ymax):
    global writer
    writer.addObject(name, xmin, ymin, xmax, ymax)

def save_anotations():
    writer.save(out_xml)
    correct_path(out_xml, image_path)

def clear_writer():
    global writer, out_xml, image_path
    writer = ''
    out_xml = ''
    image_path = ''

def correct_path(in_xml, image_path):
    tree = ET.parse(in_xml)
    root = tree.getroot()
    folder =image_path.split('/')[-2]
    # print('folder: '+ folder )
    for elem in root.iter('path'):
        elem.text = image_path
    for elem in root.iter('folder'):
        elem.text = folder
    tree.write(in_xml)
