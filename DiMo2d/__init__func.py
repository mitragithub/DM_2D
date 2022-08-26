import os
from multiprocessing import Pool
from functools import partial
from matplotlib import image as mpimg
import cv2

INDEX_STRING = '.jp2'
YEAR_STRING = '2013'


def get_date(filename):
    index = filename.index(YEAR_STRING)
    date = filename[index: index + 11]
    return date


def remove_dups(filenames):
    valid_names = {}
    for image_filename in filenames:
        num = image_filename[image_filename.index(INDEX_STRING) - 4: image_filename.index(INDEX_STRING)]
        if num not in valid_names:
            valid_names[num] = image_filename
            continue
        current_filename = valid_names[num]
        current_date = get_date(current_filename)
        current_month = int(current_date[5:7])
        current_day = int(current_date[8:10])

        image_date = get_date(image_filename)
        image_month = int(image_date[5:7])
        image_day = int(image_date[8:10])

        if (image_month > current_month) or (image_month == current_month and image_day > current_day):
            valid_names[num] = image_filename

    no_dups = list(valid_names.values())
    no_dups.sort()
    return no_dups


'''
def function(image_filename):
    input_filename = os.path.join(input_dir, image_filename)
    input_filename_command = input_filename.replace('&', '\&')
    # print('working on', input_filename)
    num = image_filename[image_filename.index(INDEX_STRING) - 4: image_filename.index(INDEX_STRING)]
    # print('working on', num, 'out of', len(image_filenames))
    kdu_command = "kdu_expand -i " + input_filename_command + " -o " + output_dir + num + ".tif -num_threads 16"
    os.system(kdu_command)
'''


def cshl_jp2_to_tif(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    image_filenames = [listing for listing in os.listdir(input_dir) if listing != 'Likelihood']
    #print(len(image_filenames))
    image_filenames = remove_dups(image_filenames)
    #print(len(image_filenames))

    for image_filename in image_filenames:
        input_filename = os.path.join(input_dir, image_filename)
        input_filename_command = input_filename.replace('&', '\&')
        # print('working on', input_filename)
        num = image_filename[image_filename.index(INDEX_STRING) - 4: image_filename.index(INDEX_STRING)]
        # print('working on', num, 'out of', len(image_filenames))
        kdu_command = "kdu_expand -i " + input_filename_command + " -o " + output_dir + num + ".tif -num_threads 16"
        os.system(kdu_command)


def split_tif_channels(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    red_dir = os.path.join(output_dir, 'red/')
    if not os.path.exists(red_dir):
        os.mkdir(red_dir)

    green_dir = os.path.join(output_dir, 'green/')
    if not os.path.exists(green_dir):
        os.mkdir(green_dir)

    image_filenames = [listing for listing in os.listdir(input_dir)]
    image_filenames.sort()

    for image_filename in image_filenames:
        input_path = os.path.join(input_dir, image_filename)
        image = mpimg.imread(input_path)
        red_channel = image[:, :, 2]
        green_channel = image[:, :, 1]
        red_output = os.path.join(red_dir, image_filename)
        green_output = os.path.join(green_dir, image_filename)
        cv2.imwrite(red_output, red_channel)
        cv2.imwrite(green_output, green_channel)


def single_crop(output_dir, input_filename):
    return


def crop_single_channel(input_dir, output_dir, threads=1):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    image_filenames = [listing for listing in os.listdir(input_dir)]
    pool = Pool(threads)
    pool.map(function, image_filenames)
