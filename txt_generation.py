# -*- coding: utf-8 -*-
import time
start = time.time()
import cv2
import argparse
import os
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import pandas as pd
import random
import pprint
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image',
	help='path to input template image')
ap.add_argument('-n', '--number',
	help='number of image to generate')
ap.add_argument('-f', '--folder',
    help='path to input template images')
args = vars(ap.parse_args())

main_dir = str(os.path.dirname(__file__)) + '/'
pp = pprint.PrettyPrinter(indent=4)

def load_names():
    female_df = pd.read_csv(main_dir+'female_names.csv',
                            # index_col='name',
                            encoding='utf-8',
                            usecols=['name']).sample(frac=1)
    male_df = pd.read_csv(main_dir+'male_names.csv',
                          # index_col='frequency',
                          encoding='utf-8',
                          usecols=['name']).sample(frac=1)
    names = pd.concat([female_df, male_df])
    lastName_df = pd.read_csv(main_dir+'surnames_freq_ge_100.csv',
                              # index_col='surname',
                              encoding='utf-8',
                              usecols=['surname']).sample(frac=1)
    return names['name'].tolist(), lastName_df['surname'].tolist()

names,  lastnames= load_names()

def generate_id_number():
    if bool(random.getrandbits(1)):
        num = int(random.gauss(1100000000,100000000)) #Gaussian distribution for number new ID
    else:
        num = int(random.gauss(5000000,4000000)) #Gaussian distribution for number for old Id
    return '{:,}'.format(num).replace(',','.')

def generate_full_name():
    name = names[random.randint(0, len(names)-1)]
    lastname = lastnames[random.randint(0, len(lastnames)-1)] + \
            ' '+lastnames[random.randint(0, len(lastnames)-1)]
    return name, lastname

def write_text_on_image(img,
                        txt,
                        bottomLeftCornerOfText = (10, 500)):
    # Write some Text
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    fontScale = 2.7
    fontColor = (0,0,0)
    lineType  = 9

    cv2.putText(img,
                txt,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    return img;

def write_text_on_image_custom_font(img,
                                    txts = [{"txt": "null",
                                            "bottomLeftCornerOfText": (10, 500),
                                            "fontColor": (55,55,55,0),
                                            "fontSize": 83}]):
    # Write some Text
    fontpath = main_dir+"Helvetica Bold.ttf"
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    for info_txt in txts:
        font = ImageFont.truetype(fontpath, info_txt["fontSize"])
        draw.text(info_txt["bottomLeftCornerOfText"],
                  info_txt["txt"],
                  font = font,
                  fill = info_txt["fontColor"])
    img = np.array(img_pil)
    return img;

def deform_id(img_path):
    iter= int(args['number']) if args['number'] != None else 1
    base_name = img_path.split('/')[-1]
    image = cv2.imread(img_path)
    sub_folder = 'output/'
    for i in range(iter):
        name, lastname = generate_full_name()
        txts = [{"txt": generate_id_number(),
                "bottomLeftCornerOfText": (390, 405),
                "fontColor": (55,55,55,0),
                "fontSize": 83},
                {"txt": lastname,
                        "bottomLeftCornerOfText": (154, 510),
                        "fontColor": (55,55,55,0),
                        "fontSize": 65},
                {"txt": name,
                        "bottomLeftCornerOfText": (148, 698),
                        "fontColor": (55,55,55,0),
                        "fontSize": 65}]
        image_out = write_text_on_image_custom_font(image, txts)

        print('Saving: '+main_dir+sub_folder+str(i+1)+'_{}.png'.format(base_name.split('.')[0]))
        cv2.imwrite(main_dir+sub_folder+str(i+1)+'_{}.png'.format(base_name.split('.')[0]), image_out)

if __name__ == '__main__':
    if args['image'] != None:
        deform_id(args['image'])

    elif args['folder'] != None:
        files = os.listdir(args['folder'])
        for f in files:
            deform_id(f)
    else:
        print('invalid arguments')
    print('Running time: ', time.time() - start, ' seconds')
