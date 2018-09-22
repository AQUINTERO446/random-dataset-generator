#docker run --rm -it -v D:\Informacion\Documents\Repositorios\Id_random_generation:/home ocr_server2 python /home/txt_generation.py -i /home/colombian_id.png -n 1000
# -*- coding: utf-8 -*-
import time
import copy
start = time.time()
import cv2
import argparse
import shutil
import os
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import pandas as pd
import random
import anotations_maker as am
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image',
                help='path to input template image')
ap.add_argument('-n', '--number',
                type=int,
                help='number of image to generate')
ap.add_argument('-f', '--folder',
                help='path to input template images')
args = vars(ap.parse_args())

main_dir = str(os.path.dirname(__file__)) + '/'

LOCAL_CLASS = {'cedula':[(160/4, 410/4),(923/4, 410/4), (923/4, 478/4), (160/4, 478/4)],
               'apellidos':[(150/4, 510/4),(1094/4, 510/4), (1094/4, 570/4), (150/4, 570/4)],
               'nombres':[(144/4, 695/4),(1094/4, 695/4), (1094/4, 755/4), (144/4, 755/4)]
               }

def load_names():
    female_df = pd.read_csv(main_dir+'female_names.csv',
                            # index_col='name',
                            encoding='utf-8',
                            usecols=['name'])
    male_df = pd.read_csv(main_dir+'male_names.csv',
                          # index_col='frequency',
                          encoding='utf-8',
                          usecols=['name'])
    names = pd.concat([female_df, male_df])
    lastName_df = pd.read_csv(main_dir+'surnames_freq_ge_100.csv',
                              # index_col='surname',
                              encoding='utf-8',
                              usecols=['surname'])
    return names['name'].tolist(), lastName_df['surname'].tolist()

names,  lastnames= load_names()

def transform(img, pt1, pt2= False):
    x,y,h,w = pt1
    if not pt2:
        xt, yt, ht, wt = pt2
    else:
        xt = 0
        yt = 0
        ht = pt1[2]
        wt= pt1[3]

    pts1 = np.float32([[y,x], [y+h,x], [y,x+w], [y+h,x+w]])
    pts2 = np.float32([[yt,xt],[yt+ht,xt],[yt,xt+wt],[yt+ht,xt+wt]])
    dst = cv2.warpPerspective(img,M,(w,h))

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def generate_id_number():
    if bool(random.getrandbits(1)):
        num = random.randint(1000000000,1100000000) #Gaussian distribution for number new ID
    else:
        num = int(random.gauss(5000000,400000)) #Gaussian distribution for number for old Id
    return '{:,}'.format(num).replace(',','.')

def generate_full_name():
    name = random.choice(names)
    lastname = random.choice(lastnames) + \
            ' '+ random.choice(lastnames)
    return name, lastname

def noisy(noise_typ,image):
    """
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.

    """
    if noise_typ == "gauss":
        row,col= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out

def aply_deformation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 4-bit Quantization
    bins = np.zeros(shape=(17))
    for i in range(17):
        bins[i] = i*16
    gray = np.digitize(gray,bins)
    gray = gray*16
    gray[gray==256] = gray[gray==256] - 1
    gray = gray.astype(np.float32) #float32 for gamma correction to be aplied

    #Gamma Correction
    #random gamma to change light conditions
    g = round(random.uniform(0.1, 0.80), 2) if bool(random.getrandbits(1)) else random.randint(3, 6)
    gray = (255**(1-g))*(gray**g)

    noises=["gauss", "s&p"]
    iter = random.randint(1,2)
    for i in range(iter):
        noise = random.choice(noises)
        noises.remove(noise)
        gray = noisy(noise, gray)
    out_image = rotate_bound(gray.astype(np.uint8), random.randint(0, 180))
    return out_image

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

def transform_point(point, M):
    # Prepare the vector to be transformed
    v = [point[0],point[1],1]
    calculated = np.dot(M,v)
    return (int(calculated[0]),int(calculated[1]))

def rotate_box(bb, M):
    new_bb = list(bb)
    for i,point in enumerate(bb):
        new_bb[i] = transform_point(point, M)
    return new_bb

def rotate_bound(image, angle):
    global LOCAL_CLASS
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    img = cv2.warpAffine(image, M, (nW, nH))

    #bounding box
    print(LOCAL_CLASS)
    for clases, coordinates in LOCAL_CLASS.items():
        box =rotate_box(coordinates, M)
        LOCAL_CLASS[clases] = box
        # cv2.line(img, box[0], box[3], 0, 2)
        # cv2.line(img, box[0], box[1], 0, 2)
        # cv2.line(img, box[1], box[2], 0, 2)
        # cv2.line(img, box[2], box[3], 0, 2)
    print(LOCAL_CLASS)
    return img

def deform_id(img_path):
    iter= int(args['number']) if args['number'] != None else 1
    base_name = img_path.split('/')[-1]
    image = cv2.imread(img_path)
    image = cv2.resize(image,(537,330))
    sub_folder = 'output'
    for i in range(iter):
        name, lastname = generate_full_name()
        txts = [{"txt": generate_id_number(),
                "bottomLeftCornerOfText": (390/4, 405/4),
                "fontColor": (55,55,55,0),
                "fontSize": int(83/4)},
                {"txt": lastname,
                        "bottomLeftCornerOfText": (154/4, 510/4),
                        "fontColor": (55,55,55,0),
                        "fontSize": int(65/4)},
                {"txt": name,
                        "bottomLeftCornerOfText": (148/4, 698/4),
                        "fontColor": (55,55,55,0),
                        "fontSize": int(65/4)}]
        image_out = aply_deformation(write_text_on_image_custom_font(image, txts))
        image_path = main_dir+sub_folder+'/'+str(i+1)+'_{}.png'.format(base_name.split('.')[0])
        print('Making anotations for ML')
        h, w = image_out.shape[:2]
        am.set_writer(image_path, h,w,)
        for clases, coordinates in LOCAL_CLASS.items():
            box = np.array(LOCAL_CLASS[clases])
            am.label_image(clases,
                           np.min(box[:,0]),
                           np.min(box[:,1]),
                           np.max(box[:,0]),
                           np.max(box[:,1]))
        am.save_anotations()
        am.clear_writer()
        print('Saving: '+image_path)
        cv2.imwrite(image_path, image_out)

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
