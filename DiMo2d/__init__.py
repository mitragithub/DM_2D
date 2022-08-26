import os
from multiprocessing import Pool
from functools import partial
from matplotlib import image as mpimg
import cv2
import numpy as np
import csv
from geojson import Feature, FeatureCollection, LineString
import geojson as gjson
from math import fabs
from shutil import copyfile

from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def get_date(filename):
    index = filename.index('2013')
    date = filename[index: index + 11]
    return date


def __remove_dups(filenames):
    valid_names = {}
    for image_filename in filenames:
        num = image_filename[image_filename.index('.jp2') - 4: image_filename.index('.jp2')]
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


def __single_jp2_to_tif(jp2_dir, tif_dir, image_filename):
    input_filename = os.path.join(jp2_dir, image_filename)
    input_filename_command = input_filename.replace('&', '\&')
    # print('working on', input_filename)
    num = image_filename[image_filename.index('.jp2') - 4: image_filename.index('.jp2')]
    # print('working on', num, 'out of', len(image_filenames))
    kdu_command = "kdu_expand -i " + input_filename_command + " -o " + tif_dir + num + ".tif -num_threads 16"
    os.system(kdu_command)


def cshl_jp2_to_tif(input_dir, output_dir, threads=1):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    image_filenames = [listing for listing in os.listdir(input_dir) if listing != 'Likelihood']
    # print(len(image_filenames))
    image_filenames = __remove_dups(image_filenames)
    # print(len(image_filenames))

    pool = Pool(threads)
    pool.map(partial(__single_jp2_to_tif, input_dir, output_dir), image_filenames)
    pool.close()
    pool.join()


def __single_split_tif_channels(input_dir, output_dir, image_filename):
    input_path = os.path.join(input_dir, image_filename)
    image = mpimg.imread(input_path)
    red_channel = image[:, :, 2]
    green_channel = image[:, :, 1]

    red_dir = os.path.join(output_dir, 'red/')
    green_dir = os.path.join(output_dir, 'green/')

    assert(os.path.exists(red_dir) and os.path.exists(green_dir))

    red_output = os.path.join(red_dir, image_filename)
    green_output = os.path.join(green_dir, image_filename)
    cv2.imwrite(red_output, red_channel)
    cv2.imwrite(green_output, green_channel)


def split_tif_channels(input_dir, output_dir, threads=1):

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

        pool = Pool(threads)
        pool.map(partial(__single_jp2_to_tif, input_dir, output_dir), image_filenames)
        pool.close()
        pool.join()


def __single_crop_channel(channel_dir, crop_dir, image_filename):
    input_filename = os.path.join(channel_dir, image_filename)
    image_output_dir = os.path.join(crop_dir, os.path.splitext(image_filename)[0]) + '/'
    # print(image_output_dir)
    crop_filename = os.path.join(image_output_dir, 'crop.txt')
    cropped_filename = os.path.join(image_output_dir, 'image.tif')
    # print(cropped_filename)

    image = mpimg.imread(input_filename)\
    # print('cropping')
    x, y = np.nonzero(image)

    # print('check:', len(x), len(y))
    if len(x) == len(y) == 0:
        return


    if np.max(image) == 31:
        return

    if not os.path.exists(image_output_dir):
        os.mkdir(image_output_dir)

    #print('cropping')
    xl, xr = x.min(), x.max()
    yl, yr = y.min(), y.max()
    cropped = image[xl - 1:xr + 2, yl - 1:yr + 2]

    #print('outputting')
    cv2.imwrite(cropped_filename, cropped)

    with open(crop_filename, 'w') as crop_file:
        crop_file.write(str(xl - 1) + ' ' + str(xr + 2) + ' ' + str(yl - 1) + ' ' + str(yr + 2) + '\n')
        crop_file.close()


def crop_channel(channel_dir, crop_dir, threads=1):
    if not os.path.exists(crop_dir):
        os.mkdir(crop_dir)

    image_filenames = [listing for listing in os.listdir(channel_dir)]
    pool = Pool(threads)
    pool.map(partial(__single_crop_channel, channel_dir, crop_dir), image_filenames)
    pool.close()
    pool.join()


def __single_write_dipha_input_file(cropped_dir):
    input_filename = os.path.join(cropped_dir, 'image.tif')
    dipha_output_filename = os.path.join(cropped_dir, 'dipha.input')
    os.system("matlab -nosplash -nodisplay -nodesktop -r \'save_image_data(\"" + input_filename + "\",\"" + dipha_output_filename + "\"); quit;\'")


def write_dipha_input_files(input_dir, threads=1):

    cropped_image_dirs = [os.path.join(input_dir, listing) for listing in os.listdir(input_dir)]
    cropped_image_dirs.sort()

    pool = Pool(threads)
    pool.map(__single_write_dipha_input_file, cropped_image_dirs)
    pool.close()
    pool.join()


def __single_write_vertex_file(cropped_dir):
    input_filename = os.path.join(cropped_dir, 'image.tif')
    vert_filename = os.path.join(cropped_dir, 'vert.txt')

    image = mpimg.imread(input_filename)
    nx, ny = image.shape

    # print('writing vert file')
    with open(vert_filename, 'w') as vert_file:
        for j in range(ny):
            for i in range(nx):
                vert_file.write(str(i) + ' ' + str(j) + ' ' + str(image[i, j]) + '\n')
        vert_file.close()


def write_vertex_files(input_dir, threads=1):

    cropped_image_dirs = [os.path.join(input_dir, listing) for listing in os.listdir(input_dir)]
    cropped_image_dirs.sort()

    pool = Pool(threads)
    pool.map(__single_write_vertex_file, cropped_image_dirs)
    pool.close()
    pool.join()


def __single_run_dipha_persistence(cropped_dir, mpi_threads=1):
    input_filename = os.path.join(cropped_dir, 'image.tif')
    dipha_output_filename = os.path.join(cropped_dir, 'dipha.input')
    diagram_filename = os.path.join(cropped_dir, 'diagram.bin')
    dipha_edge_filename = os.path.join(cropped_dir, 'dipha-thresh.edges')

    image = mpimg.imread(input_filename)
    nx, ny = image.shape

    command = 'mpiexec -n ' + str(
        mpi_threads) + ' ./DiMo2d/code/dipha-2d-thresh/build/dipha --upper_dim 2 ' + dipha_output_filename + ' ' + diagram_filename + ' ' + dipha_edge_filename + ' ' + str(
        nx) + ' ' + str(ny)
    # command = 'mpiexec -n 32 ../dipha/dipha-graph-recon/build/dipha --upper_dim 2 dipha/neuron1-smaller/complex.bin dipha/neuron1-smaller/persistence.diagram dipha/neuron1-smaller/dipha.edges 235 248 251'
    os.system(command)


def run_dipha_persistence(input_dir, threads=1):

    #print(input_dir)
    cropped_image_dirs = [os.path.join(input_dir, listing) for listing in os.listdir(input_dir)]
    cropped_image_dirs.sort()
    #print(cropped_image_dirs)

    pool = Pool(threads)
    pool.map(partial(__single_run_dipha_persistence), cropped_image_dirs)
    pool.close()
    pool.join()


def __single_convert_persistence_diagrams(cropped_dir):
    dipha_edge_filename = os.path.join(cropped_dir, 'dipha-thresh.edges')
    dimo_input_filename = os.path.join(cropped_dir, 'dipha-edges.txt')
    matlab_command = "matlab -r 'load_persistence_diagram(" + '"' + dipha_edge_filename + '", "' + dimo_input_filename + '"); exit;' + "'"
    os.system(matlab_command)


def convert_persistence_diagrams(input_dir, threads=1):
    cropped_image_dirs = [os.path.join(input_dir, listing) for listing in os.listdir(input_dir)]
    cropped_image_dirs.sort()

    pool = Pool(threads)
    pool.map(__single_convert_persistence_diagrams, cropped_image_dirs)
    pool.close()
    pool.join()


def compute_persistence_single_channel(input_dir, output_dir, threads=1):
    crop_channel(input_dir, output_dir, threads)
    write_dipha_input_files(output_dir, threads)
    run_dipha_persistence(output_dir, threads)
    convert_persistence_diagrams(output_dir, threads)
    write_vertex_files(output_dir, threads)


def __single_graph_reconstruction(ve_persistence_threshold, et_persistence_threshold, cropped_dir):
    dimo_input_filename = os.path.join(cropped_dir, 'dipha-edges.txt')
    vert_filename = os.path.join(cropped_dir, 'vert.txt')
    dimo_output_dir = os.path.join(cropped_dir, str(ve_persistence_threshold) + '_' + str(et_persistence_threshold) + '/')

    if not os.path.exists(dimo_output_dir):
        os.mkdir(dimo_output_dir)

    morse_command = './DiMo2d/code/dipha-output-2d-ve-et-thresh/a.out ' + vert_filename + ' ' + dimo_input_filename + ' ' \
                    + str(ve_persistence_threshold) + ' ' + str(et_persistence_threshold) + ' ' + dimo_output_dir
    # print(morse_command)
    os.system(morse_command)


def run_graph_reconstruction(input_dir, ve_persistence_threshold=0, et_persistence_threshold=64, threads=1):

    cropped_image_dirs = [os.path.join(input_dir, listing) for listing in os.listdir(input_dir)]
    cropped_image_dirs.sort()

    pool = Pool(threads)
    pool.map(partial(__single_graph_reconstruction, ve_persistence_threshold, et_persistence_threshold), cropped_image_dirs)
    pool.close()
    pool.join()


def __single_shift_vertex_coordinates(ve_persistence_threshold, et_persistence_threshold, cropped_dir):
    dimo_output_dir = os.path.join(cropped_dir, str(ve_persistence_threshold) + '_' + str(et_persistence_threshold) + '/')
    input_vert_filename = os.path.join(dimo_output_dir, 'dimo_vert.txt')
    output_vert_filename = os.path.join(dimo_output_dir, 'uncropped_dimo_vert.txt')
    crop_filename = os.path.join(cropped_dir, 'crop.txt')

    with open(crop_filename, 'r') as crop_file:
        reader = csv.reader(crop_file, delimiter=' ')
        for row in reader:
            x_add = int(row[0])
            y_add = int(row[2])
            break
        crop_file.close()

    with open(output_vert_filename, 'w') as output_vert_file:
        with open(input_vert_filename, 'r') as input_vert_file:
            reader = csv.reader(input_vert_file, delimiter=' ')
            for row in reader:
                # 2 is function value, need for vector
                output_vert_file.write(
                    str(int(row[0]) + x_add) + ' ' + str(int(row[1]) + y_add) + ' ' + row[2] + '\n')
            input_vert_file.close()
        output_vert_file.close()


def shift_vertex_coordinates(input_dir, ve_persistence_threshold=0, et_persistence_threshold=64, threads=1):

        cropped_image_dirs = [os.path.join(input_dir, listing) for listing in os.listdir(input_dir)]
        cropped_image_dirs.sort()

        pool = Pool(threads)
        pool.map(partial(__single_shift_vertex_coordinates, ve_persistence_threshold, et_persistence_threshold), cropped_image_dirs)
        pool.close()
        pool.join()


def __single_intersect_morse_graph_with_binary_output(input_dir, binary_dir, ve_persistence_threshold, et_persistence_threshold, image_filename):
    image_output_dir = os.path.join(input_dir, os.path.splitext(image_filename)[0]) \
                   + '/' + str(ve_persistence_threshold) + '_' + str(et_persistence_threshold) + '/'
    binary_process = os.path.join(binary_dir, image_filename + '.tif')

    vert_filename = os.path.join(image_output_dir, 'uncropped_dimo_vert.txt')
    edge_filename = os.path.join(image_output_dir, 'dimo_edge.txt')

    crossed_vert_filename = os.path.join(image_output_dir, 'crossed-vert.txt')
    crossed_edge_filename = os.path.join(image_output_dir, 'crossed-edge.txt')

    # print('reading binary')
    binary = mpimg.imread(binary_process)

    # print('reading verts')
    verts = []
    with open(vert_filename, 'r') as vert_file:
        reader = csv.reader(vert_file, delimiter=' ')
        for row in reader:
            verts.append([int(row[0]), int(row[1]), row[2]])
        vert_file.close()

    # print('reading edges')
    edges = []
    with open(edge_filename, 'r') as edge_file:
        reader = csv.reader(edge_file, delimiter=' ')
        for row in reader:
            edges.append([int(row[0]), int(row[1])])
        edge_file.close()

    # print('checking included verts')
    vert_index_dict = {}
    v_ind = 0
    for i in range(len(verts)):
        v = verts[i]
        val = binary[v[0], v[1]]
        if val == 255:
            vert_index_dict[i] = v_ind
            v_ind += 1
            continue
        assert (val == 0)
        vert_index_dict[i] = -1

    # print('outputting verts')
    with open(crossed_vert_filename, 'w') as crossed_vert_file:
        for i in range(len(verts)):
            if vert_index_dict[i] == -1:
                continue
            v = verts[i]
            crossed_vert_file.write(str(v[0]) + ' ' + str(v[1]) + ' ' + v[2] + '\n')
        crossed_vert_file.close()

    # print('outputting edges')
    with open(crossed_edge_filename, 'w') as crossed_edge_file:
        for e in edges:
            if vert_index_dict[e[0]] == -1 or vert_index_dict[e[1]] == -1:
                continue
            crossed_edge_file.write(str(vert_index_dict[e[0]]) + ' ' + str(vert_index_dict[e[1]]) + '\n')
        crossed_edge_file.close()


def intersect_morse_graphs_with_binary_outputs(input_dir, binary_dir, ve_persistence_threshold=0, et_persistence_threshold=64, threads=1):
    image_filenames = [listing for listing in os.listdir(input_dir)]
    image_filenames.sort()

    pool = Pool(threads)
    pool.map(partial(__single_intersect_morse_graph_with_binary_output, input_dir, binary_dir, ve_persistence_threshold, et_persistence_threshold), image_filenames)
    pool.close()
    pool.join()


def __single_remove_duplicate_edges(input_dir, ve_persistence_threshold, et_persistence_threshold, image_filename):
    image_output_dir = os.path.join(input_dir, os.path.splitext(image_filename)[0]) \
                       + '/' + str(ve_persistence_threshold) + '_' + str(
        et_persistence_threshold) + '/'
    input_filename = os.path.join(image_output_dir, 'dimo_edge.txt')
    output_filename = os.path.join(image_output_dir, 'no-dup-crossed-edge.txt')

    edges = set()
    with open(input_filename, 'r') as input_file:
        reader = csv.reader(input_file, delimiter=' ')
        for row in reader:
            v0 = int(row[0])
            v1 = int(row[1])
            if v0 < v1:
                vmin = v0
                vmax = v1
            else:
                vmin = v1
                vmax = v0
            if (vmin, vmax) not in edges:
                edges.add((vmin, vmax))
        input_file.close()

    with open(output_filename, 'w') as output_file:
        for e in edges:
            output_file.write(str(e[0]) + ' ' + str(e[1]) + '\n')
        output_file.close()


def remove_duplicate_edges(input_dir, ve_persistence_threshold=0, et_persistence_threshold=64, threads=1):
    image_filenames = [listing for listing in os.listdir(input_dir)]
    image_filenames.sort()

    pool = Pool(threads)
    pool.map(partial(__single_remove_duplicate_edges, input_dir, ve_persistence_threshold, et_persistence_threshold), image_filenames)
    pool.close()
    pool.join()


def generate_morse_graphs(input_dir, binary_dir, ve_persistence_threshold=0, et_persistence_threshold=64, threads=1):
    run_graph_reconstruction(input_dir, ve_persistence_threshold, et_persistence_threshold, threads)
    shift_vertex_coordinates(input_dir, ve_persistence_threshold, et_persistence_threshold, threads)
    intersect_morse_graphs_with_binary_outputs(input_dir, binary_dir, ve_persistence_threshold, et_persistence_threshold, threads)
    remove_duplicate_edges(input_dir, ve_persistence_threshold, et_persistence_threshold, threads)


def __single_non_degree_2_paths(input_dir, ve_persistence_threshold, et_persistence_threshold, image_filename):
    image_output_dir = os.path.join(input_dir, os.path.splitext(image_filename)[0]) \
        + '/' + str(ve_persistence_threshold) + '_' + str(et_persistence_threshold) + '/'
    command = './DiMo2d/code/paths_src/a.out ' + image_output_dir
    os.system(command)


def non_degree_2_paths(input_dir, ve_persistence_threshold=0, et_persistence_threshold=64, threads=1):
    image_filenames = [listing for listing in os.listdir(input_dir)]
    image_filenames.sort()

    pool = Pool(threads)
    pool.map(partial(__single_non_degree_2_paths, input_dir, ve_persistence_threshold, et_persistence_threshold), image_filenames)
    pool.close()
    pool.join()


def __single_haircut(input_dir, ve_persistence_threshold, et_persistence_threshold, image_filename):
    image_output_dir = os.path.join(input_dir, os.path.splitext(image_filename)[0]) \
                       + '/' + str(ve_persistence_threshold) + '_' + str(et_persistence_threshold) + '/'
    # vert_filename = os.path.join(image_output_dir, 'crossed-vert.txt')
    vert_filename = os.path.join(image_output_dir, 'dimo_vert.txt')
    paths_filename = os.path.join(image_output_dir, 'paths.txt')
    # sobel_filename = os.path.join(input_dir, os.path.splitext(image_filename)[0] + '/sobel-100.tif')
    output_edge_filename = os.path.join(image_output_dir, 'haircut-edge.txt')
    print('reading verts')
    verts = []
    with open(vert_filename, 'r') as vert_file:
        reader = csv.reader(vert_file, delimiter=' ')
        for row in reader:
            verts.append([int(row[0]), int(row[1]), int(row[2])])
        vert_file.close()
    print(len(verts), 'verts')
    print('reading paths')
    with open(paths_filename, 'r') as paths_file:
        content = paths_file.readlines()
        paths_file.close()

    paths = [c.strip().split(' ') for c in content]
    paths = [[int(n) for n in c] for c in paths]

    valid_paths = []
    for p in paths:
        '''
        vals = [verts[v][2] for v in p]
        if min(vals) < 50:
            continue
        '''
        valid_paths.append(p)

    degrees = {}
    for p in valid_paths:
        if p[0] not in degrees.keys():
            degrees[p[0]] = 1
        else:
            degrees[p[0]] += 1
        if p[len(p) - 1] not in degrees.keys():
            degrees[p[len(p) - 1]] = 1
        else:
            degrees[p[len(p) - 1]] += 1

    '''
    image = mpimg.imread(sobel_filename)
    image.astype('uint16')
    '''

    with open(output_edge_filename, 'w') as output_edge_file:
        for i in range(len(valid_paths)):
            # print('path:', i)
            p = valid_paths[i]
            # print(p)
            if len(p) < 2:
                output_edge_file.write(content[i] + '\n')
                print('less than 2')
                continue
            
            # print(len(verts), p[0], p[1])

            if verts[p[0]][0] == verts[p[1]][0]:
                direction = 1
            else:
                assert (verts[p[0]][1] == verts[p[1]][1])
                direction = 0
            delta = 0
            for j in range(1, len(p)):
                if verts[p[j - 1]][0] == verts[p[j]][0]:
                    current_direction = 1
                else:
                    assert (verts[p[j - 1]][1] == verts[p[j]][1])
                    current_direction = 0
                if current_direction == direction:
                    continue
                direction = current_direction
                delta += 1

            first_degree = degrees[p[0]]
            second_degree = degrees[p[len(p) - 1]]

            # haircut
            if delta <= 1 and (first_degree == 1 or second_degree == 1) and (first_degree > 2 or second_degree > 2):
                continue

            for j in range(len(p) - 1):
                output_edge_file.write(str(p[j]) + ' ' + str(p[j + 1]) + '\n')
                # output_edge_file.write(str(p[j]) + ' ' + str(p[j + 1]) + ' ' + str(int(255 * min_func / max_func)) + '\n')
        output_edge_file.close()


def haircut(input_dir, ve_persistence_threshold=0, et_persistence_threshold=64, threads=1):
    image_filenames = [listing for listing in os.listdir(input_dir)]
    image_filenames.sort()

    pool = Pool(threads)
    pool.map(partial(__single_haircut, input_dir, ve_persistence_threshold, et_persistence_threshold), image_filenames)
    pool.close()
    pool.join()


def postprocess_graphs(input_dir, ve_persistence_threshold=0, et_persistence_threshold=64, threads=1):
    non_degree_2_paths(input_dir, ve_persistence_threshold, et_persistence_threshold)
    haircut(input_dir, ve_persistence_threshold, et_persistence_threshold)


def __single_align_coordinates_with_webviewer(input_dir, ve_persistence_threshold, et_persistence_threshold, image_filename):
    image_output_dir = os.path.join(input_dir, os.path.splitext(image_filename)[0]) \
                       + '/' + str(ve_persistence_threshold) + '_' + str(et_persistence_threshold) + '/'
    # input_filename = os.path.join(image_output_dir, 'crossed-vert.txt')
    input_filename = os.path.join(image_output_dir, 'dimo_vert.txt')
    output_filename = os.path.join(image_output_dir, 'json-vert.txt')

    x_vals = []
    y_vals = []
    with open(output_filename, 'w') as output_file:
        with open(input_filename, 'r') as input_file:
            reader = csv.reader(input_file, delimiter=' ')
            for row in reader:
                raw_y = int(row[1])
                raw_x = int(row[0])

                x = raw_y
                y = - raw_x

                x_vals.append(x)
                y_vals.append(y)

                output_file.write(str(y) + ' ' + str(x) + ' 0 0' + '\n')
            input_file.close()
        output_file.close()


def cshl_align_coordinates_with_webviewer(input_dir, ve_persistence_threshold=0, et_persistence_threshold=64, threads=1):
    image_filenames = [listing for listing in os.listdir(input_dir)]
    image_filenames.sort()

    pool = Pool(threads)
    pool.map(partial(__single_align_coordinates_with_webviewer, input_dir, ve_persistence_threshold, et_persistence_threshold), image_filenames)
    pool.close()
    pool.join()


def __read_ve(vfilename, efilename):
    VX, VY, VZ = 1, 1, 1
    nodes, edges = [], []
    with open(vfilename) as file:
        for line in file:
            node = [float(x) for x in line.strip().split()[:3]]
            node[0], node[1] = node[1] * VX, node[0] * VY
            node[2] = node[2] * VZ
            node = tuple(node)
            nodes.append(node)
    with open(efilename) as file:
        for line in file:
            edge = tuple([int(x) for x in line.strip().split()[:2]])
            edges.append(edge)
            '''
            if len(edges) % 100000 == 0:
                print(len(edges))
                sys.stdout.flush()
            '''
    return nodes, edges


def __in_between(z, uz, vz, eps=1e-6):
    max_uv = max(uz, vz)
    min_uv = min(uz, vz)
    return ((min_uv < z + 0.5)) and ((max_uv > z - 0.5))


def __segment(u, v, z):
    if fabs(u[2] - v[2]) < 1e-5:
        # return (u[0]+(u[0] - v[0])/4, u[1]+(u[1] - v[1])/4, v[0]-(u[0] - v[0])/4, v[1]-(u[1] - v[1])/4)
        return (u[0], u[1], v[0], v[1])
    z_top = z - 0.5
    z_down = z + 0.5
    if u[2] > v[2]:
        u, v = v, u
    ru, rv = list(u), list(v)
    if u[2] < z_top:
        scale = (z_top - u[2]) / (v[2] - u[2])
        ru[0] = scale * (v[0] - u[0]) + u[0]
        ru[1] = scale * (v[1] - u[1]) + u[1]
    if v[2] > z_down:
        scale = (v[2] - z_down) / (v[2] - u[2])
        rv[0] = v[0] - scale * (v[0] - u[0])
        rv[1] = v[1] - scale * (v[1] - u[1])
    # return (ru[0]+(ru[0] - rv[0])/4, ru[1]+(ru[1] - rv[1])/4, rv[0]-(ru[0] - rv[0])/4, rv[1]-(ru[1] - rv[1])/4)
    return (ru[0], ru[1], rv[0], rv[1])


def __get_all_segs(nodes, edges, z_range):
    print(len(edges), len(nodes))
    seg_all = [[] for i in range(z_range + 1)]
    max_density = 0.0
    for z in range(z_range):
        # print(z)
        # sys.stdout.flush()
        for e in range(len(edges)):
            edge = edges[e]
            u = nodes[edge[0]]
            v = nodes[edge[1]]
            if __in_between(z, u[2], v[2]):
                seg = __segment(u, v, z)
                density = 1
                # density = get_density(seg, cloud_list[z])
                seg_all[z].append((seg, density, e))  # seg = (x1, y1, x2, y2), density, id(e))
                max_density = max(max_density, density)
    return max_density, seg_all


def __make_geojson(seg_all, z_range, dir_path, max_density, ind_array=None, scale=1, max_width=10):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    if ind_array is None:
        ind_array = [i for i in range(z_range)]
    for z in range(z_range):
        features = []
        json_filename = '{:04d}.json'.format(ind_array[z - 1])
        output_file = os.path.join(dir_path, json_filename)
        for seg in seg_all[z]:
            seg_rescale = [x * scale for x in seg[0]]
            features.append(Feature(id=seg[2], geometry=LineString(
                [(seg_rescale[0], seg_rescale[1]), (seg_rescale[2], seg_rescale[3])]),
                                    properties={"stroke-width": 1}))
        with open(output_file, 'w') as file:
            file.write(gjson.dumps(FeatureCollection(features), sort_keys=True))


def __single_convert_morse_graphs_to_geojson(input_dir, ve_persistence_threshold, et_persistence_threshold, image_filename):
    image_output_dir = os.path.join(input_dir, os.path.splitext(image_filename)[0]) \
                       + '/' + str(ve_persistence_threshold) + '_' + str(et_persistence_threshold) + '/'

    # flag_image = True
    # cloud_file = ''

    file_vert = os.path.join(image_output_dir, 'json-vert.txt')
    file_edge = os.path.join(image_output_dir, 'haircut-edge.txt')
    dir_name = os.path.dirname(file_vert)
    z_range = 1
    length, width = 24000, 24000
    # print(file_vert, file_edge, z_range, length, width)
    # sys.stdout.flush()
    nodes, edges = __read_ve(file_vert, file_edge)

    # print('Get segments')
    # sys.stdout.flush()
    max_density, seg_all = __get_all_segs(nodes, edges, z_range)
    output_json = os.path.join(dir_name, 'GeoJson')
    __make_geojson(seg_all, z_range, output_json, max_density)


def convert_morse_graphs_to_geojson(input_dir, ve_persistence_threshold=0, et_persistence_threshold=64, threads=1):

    image_filenames = [listing for listing in os.listdir(input_dir)]
    image_filenames.sort()

    pool = Pool(threads)
    pool.map(partial(__single_convert_morse_graphs_to_geojson, input_dir, ve_persistence_threshold, et_persistence_threshold), image_filenames)
    pool.close()
    pool.join()


def __single_move_geojson_to_folder(input_dir, output_dir, ve_persistence_threshold, et_persistence_threshold, image_filename):
    image_output_dir = os.path.join(input_dir, os.path.splitext(image_filename)[0]) \
                       + '/' + str(ve_persistence_threshold) + '_' + str(et_persistence_threshold) + '/'
    json_filename = os.path.join(image_output_dir, 'GeoJson/0000.json')

    output_filename = os.path.join(output_dir, os.path.splitext(image_filename)[0] + '.json')
    copyfile(json_filename, output_filename)


def move_geojsons_to_folder(input_dir, output_dir, ve_persistence_threshold=0, et_persistence_threshold=64, threads=1):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    image_filenames = [listing for listing in os.listdir(input_dir)]
    image_filenames.sort()

    pool = Pool(threads)
    pool.map(partial(__single_move_geojson_to_folder, input_dir, output_dir, ve_persistence_threshold, et_persistence_threshold), image_filenames)
    pool.close()
    pool.join()


def cshl_post_results(input_dir, output_dir, ve_persistence_threshold=0, et_persistence_threshold=64, threads=1):
    cshl_align_coordinates_with_webviewer(input_dir, ve_persistence_threshold, et_persistence_threshold, threads)
    convert_morse_graphs_to_geojson(input_dir, ve_persistence_threshold, et_persistence_threshold, threads)
    move_geojsons_to_folder(input_dir, output_dir, ve_persistence_threshold, et_persistence_threshold, threads)

