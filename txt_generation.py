import cv2
import argparse
import os
from PIL import ImageFont, ImageDraw, Image
import numpy as np
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image',
	help='path to input image to be OCRd')
ap.add_argument('-f', '--folder',
    help='path to input images to be OCRd')
args = vars(ap.parse_args())

def write_text_on_image(img, txt, bottomLeftCornerOfText = (10, 500)):
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

def write_text_on_image_custom_font(img, txt, bottomLeftCornerOfText = (10, 500)):
    # Write some Text
    fontpath = main_dir+"Helvetica Bold.ttf"
    font = ImageFont.truetype(fontpath, 82)
    img_pil = Image.fromarray(img)
    fontColor = (0,0,0,0)
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
                                '1.098.777.900',
                                bottomLeftCornerOfText = (390, 405))
    print('Saving: '+main_dir+'write_{}.png'.format(base_name.split('.')[0]))
    cv2.imwrite(main_dir+'write_{}.png'.format(base_name.split('.')[0]), image)

if __name__ == '__main__':
    main_dir = str(os.path.dirname(__file__)) + '/'
    if args['image'] != None:
        deform_id(args['image'])

    elif args['folder'] != None:
        files = os.listdir(args['folder'])
        for f in files:
            str[f] = deform_id(f)
    else:
        print('Missing arguments: --image or --folder')
