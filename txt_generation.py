# -*- coding: utf-8 -*-
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
	help='path to input image to be OCRd')
ap.add_argument('-f', '--folder',
    help='path to input images to be OCRd')
args = vars(ap.parse_args())

main_dir = str(os.path.dirname(__file__)) + '/'
pp = pprint.PrettyPrinter(indent=4)

female_df = pd.read_csv(main_dir+'female_names.csv',  index_col='name', encoding='utf-8')
male_df = pd.read_csv(main_dir+'male_names.csv',  index_col='name', encoding='utf-8')
lastName_df = pd.read_csv(main_dir+'surnames_freq_ge_100.csv',  index_col='surname', encoding='utf-8')

def generate_id_number():
    if bool(random.getrandbits(1)):
        num = int(random.gauss(1100000000,100000000)) #Gaussian distribution for number new ID
    else:
        num = int(random.gauss(5000000,4000000)) #Gaussian distribution for number for old Id
    return '{:,}'.format(num).replace(',','.')

def generate_full_name():
    pp.pprint(female_df.head())
    pp.pprint(male_df.head())
    pp.pprint(lastName_df.head(20))


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
                                    txt,
                                    bottomLeftCornerOfText = (10, 500),
                                    fontColor = (55,55,55,0),
                                    fontSize = 83):
    # Write some Text
    fontpath = main_dir+"Helvetica Bold.ttf"
    font = ImageFont.truetype(fontpath, fontSize)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(bottomLeftCornerOfText,
              txt,
              font = font,
              fill = fontColor)
    img = np.array(img_pil)
    return img;

def deform_id(img_path):
    base_name = img_path.split('/')[-1]
    image = cv2.imread(img_path)
    image = write_text_on_image_custom_font(image,
                                generate_id_number(),
                                (390, 405),
                                (55,55,55,0),
                                83)
    image = write_text_on_image_custom_font(image,
                                'QUINTERO GALVAN',
                                (154, 510),
                                (55,55,55,0),
                                65)
    image = write_text_on_image_custom_font(image,
                                'ANDRES',
                                (148, 698),
                                (55,55,55,0),
                                65)

    print('Saving: '+main_dir+'write_{}.png'.format(base_name.split('.')[0]))
    cv2.imwrite(main_dir+'write_{}.png'.format(base_name.split('.')[0]), image)

if __name__ == '__main__':
    if args['image'] != None:
        deform_id(args['image'])

    elif args['folder'] != None:
        files = os.listdir(args['folder'])
        for f in files:
            str[f] = deform_id(f)
    else:
        generate_full_name()
