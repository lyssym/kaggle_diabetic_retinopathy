# coding: utf-8

import pandas as p
import os
import PIL
from PIL import ImageEnhance
from PIL import Image, ImageChops, ImageOps
import numpy as np

from utils import get_img_ids_from_iter, get_img_ids_from_dir
from utils import split_data, softmax, oversample_set
from config import default_transfo_params, batch_size, height, width

train_labels = p.read_csv('data/trainLabels.csv')
all_train_patient_ids = set(get_img_ids_from_iter(train_labels.image))


def make_thumb(image, size=(80, 80), pad=False):
    image.thumbnail(size, Image.BILINEAR)
    image_size = image.size

    if pad:
        thumb = image.crop((0, 0, size[0], size[1]))

        offset_x = max((size[0] - image_size[0]) / 2, 0)
        offset_y = max((size[1] - image_size[1]) / 2, 0)

        thumb = ImageChops.offset(thumb, offset_x, offset_y)
    else:
        thumb = ImageOps.fit(image, size, Image.BILINEAR, (0.5, 0.5))

    return thumb


def load_image_and_process(im, im_dst,flag,
                           output_shape=(512, 512),
                           prefix_path='train_raw/',
                           transfo_params=None,
                           rand_values=None):
    # target = prefix_path + im + '.jpeg'
    target = prefix_path + im
    im_default = np.zeros((width, height, 3), dtype='float32')
    if not os.path.exists(target):
        im_dst[:] = im_default
        flag = 1
        return

    flag = 0
    im = Image.open(target, mode='r')
    im_new = im
    # im_new.show()
    chosen_values = {}

    if transfo_params.get('extra_width_crop', False):
        w, h = im_new.size

        if w / float(h) >= 1.3:
            cols_thres = np.where(
                np.max(
                    np.max(
                        np.asarray(im_new),
                        axis=2),
                    axis=0) > 35)[0]

            # Extra cond compared to orig crop.
            if len(cols_thres) > output_shape[0] // 2:
                min_x, max_x = cols_thres[0], cols_thres[-1]
            else:
                min_x, max_x = 0, -1

            im_new = im_new.crop((min_x, 0,
                                  max_x, h))

    if transfo_params.get('crop_height', False):
        w, h = im_new.size

        if w > 1 and 0.98 <= h / float(w) <= 1.02:
            # "Normal" without height crop, do height crop.
            im_new = im_new.crop((0, int(0.05 * h),
                                  w, int(0.95 * h)))

    if transfo_params.get('crop', False) and not \
            transfo_params.get('crop_after_rotation', False):
        if rand_values:
            do_crop = rand_values['do_crop']
        else:
            do_crop = transfo_params['crop_prob'] > np.random.rand()
        chosen_values['do_crop'] = do_crop

        if do_crop:
            out_w, out_h = im_new.size
            w_dev = int(transfo_params['crop_w'] * out_w)
            h_dev = int(transfo_params['crop_h'] * out_h)

            # If values are supplied.
            if rand_values:
                w0, w1 = rand_values['w0'], rand_values['w1']
                h0, h1 = rand_values['h0'], rand_values['h1']
            else:
                w0 = np.random.randint(0, w_dev + 1)
                w1 = np.random.randint(0, w_dev + 1)
                h0 = np.random.randint(0, h_dev + 1)
                h1 = np.random.randint(0, h_dev + 1)

            # Add params to dict.
            chosen_values['w0'] = w0
            chosen_values['w1'] = w1
            chosen_values['h0'] = h0
            chosen_values['h1'] = h1

            im_new = im_new.crop((0 + w0, 0 + h0,
                                  out_w - w1, out_h - h1))

    if transfo_params.get('shear', False):
        if transfo_params['shear_prob'] > np.random.rand():
            # print 'shear'
            # TODO: No chosen values because shear not really used.
            shear_min, shear_max = transfo_params['shear_range']
            m = shear_min + np.random.rand() * (shear_max - shear_min)
            out_w, out_h = im_new.size
            xshift = abs(m) * out_w
            new_width = out_w + int(round(xshift))
            im_new = im_new.transform((new_width, out_h), Image.AFFINE,
                                      (1, m, -xshift if m > 0 else 0, 0, 1, 0),
                                      Image.BICUBIC)

    if transfo_params.get('rotation_before_resize', False):
        if rand_values:
            rotation_param = rand_values['rotation_param']
        else:
            rotation_param = np.random.randint(
                transfo_params['rotation_range'][0],
                transfo_params['rotation_range'][1])
        chosen_values['rotation_param'] = rotation_param

        im_new = im_new.rotate(rotation_param, resample=Image.BILINEAR,
                               expand=transfo_params.get('rotation_expand',
                                                         False))
        if transfo_params.get('rotation_expand', False):
            im_new = im_new.crop(im_new.getbbox())

    if transfo_params.get('crop_after_rotation', False):
        if rand_values:
            do_crop = rand_values['do_crop']
        else:
            do_crop = transfo_params['crop_prob'] > np.random.rand()
        chosen_values['do_crop'] = do_crop

        if do_crop:
            out_w, out_h = im_new.size
            w_dev = int(transfo_params['crop_w'] * out_w)
            h_dev = int(transfo_params['crop_h'] * out_h)

            # If values are supplied.
            if rand_values:
                w0, w1 = rand_values['w0'], rand_values['w1']
                h0, h1 = rand_values['h0'], rand_values['h1']
            else:
                w0 = np.random.randint(0, w_dev + 1)
                w1 = np.random.randint(0, w_dev + 1)
                h0 = np.random.randint(0, h_dev + 1)
                h1 = np.random.randint(0, h_dev + 1)

            # Add params to dict.
            chosen_values['w0'] = w0
            chosen_values['w1'] = w1
            chosen_values['h0'] = h0
            chosen_values['h1'] = h1

            im_new = im_new.crop((0 + w0, 0 + h0,
                                  out_w - w1, out_h - h1))

    if transfo_params.get('keep_aspect_ratio', False):
        im_new = make_thumb(im_new, size=output_shape,
                           pad=transfo_params['resize_pad'])
    else:
        im_new = im_new.resize(output_shape, resample=Image.BILINEAR)

    if transfo_params.get('rotation', False) \
            and not transfo_params.get('rotation_before_resize', False):
        if rand_values:
            rotation_param = rand_values['rotation_param']
        else:
            rotation_param = np.random.randint(
                transfo_params['rotation_range'][0],
                transfo_params['rotation_range'][1])
        chosen_values['rotation_param'] = rotation_param

        im_new = im_new.rotate(rotation_param, resample=Image.BILINEAR,
                               expand=transfo_params.get('rotation_expand',
                                                         False))
        if transfo_params.get('rotation_expand',
                              False):
            im_new = im_new.crop(im_new.getbbox())

    if transfo_params.get('contrast', False):
        contrast_min, contrast_max = transfo_params['contrast_range']
        if rand_values:
            contrast_param = rand_values['contrast_param']
        else:
            contrast_param = np.random.uniform(contrast_min, contrast_max)
        chosen_values['contrast_param'] = contrast_param

        im_new = ImageEnhance.Contrast(im_new).enhance(contrast_param)

    if transfo_params.get('brightness', False):
        brightness_min, brightness_max = transfo_params['brightness_range']
        if rand_values:
            brightness_param = rand_values['brightness_param']
        else:
            brightness_param = np.random.uniform(brightness_min,
                                                 brightness_max)
        chosen_values['brightness_param'] = brightness_param

        im_new = ImageEnhance.Brightness(im_new).enhance(brightness_param)

    if transfo_params.get('color', False):
        color_min, color_max = transfo_params['color_range']
        if rand_values:
            color_param = rand_values['color_param']
        else:
            color_param = np.random.uniform(color_min, color_max)
        chosen_values['color_param'] = color_param

        im_new = ImageEnhance.Color(im_new).enhance(color_param)

    if transfo_params.get('flip', False):
        if rand_values:
            do_flip = rand_values['do_flip']
        else:
            do_flip = transfo_params['flip_prob'] > np.random.rand()

        chosen_values['do_flip'] = do_flip

        if do_flip:
            im_new = im_new.transpose(Image.FLIP_LEFT_RIGHT)

    if output_shape[0] < 200 and False:
        # Otherwise too slow.
        # TODO: Disabled for now
        if 'rotation' in transfo_params and transfo_params['rotation']:
            if rand_values:
                rotation_param = rand_values['rotation_param2']
            else:
                rotation_param = np.random.randint(
                    transfo_params['rotation_range'][0],
                    transfo_params['rotation_range'][1])

            im_new = im_new.rotate(rotation_param, resample=Image.BILINEAR,
                                   expand=False)

            chosen_values['rotation_param2'] = rotation_param

    if transfo_params.get('zoom', False):
        if rand_values:
            do_zoom = rand_values['do_zoom']
        else:
            do_zoom = transfo_params['zoom_prob'] > np.random.rand()
        chosen_values['do_zoom'] = do_zoom

        if do_zoom:
            zoom_min, zoom_max = transfo_params['zoom_range']
            out_w, out_h = im_new.size
            if rand_values:
                w_dev = rand_values['w_dev']
            else:
                w_dev = int(np.random.uniform(zoom_min, zoom_max) / 2 * out_w)
            chosen_values['w_dev'] = w_dev

            im_new = im_new.crop((0 + w_dev,
                                  0 + w_dev,
                                  out_w - w_dev,
                                  out_h - w_dev))

    if im_new.size != output_shape:
        im_new = im_new.resize(output_shape, resample=Image.BILINEAR)

    im_new.show()
    im_new = np.asarray(im_new).astype('float32') / 255
    im_dst[:] = im_new
    im.close()
    del im, im_new
    return chosen_values


def patches_gen_pairs(images, labels,
                      weight=512, height=512,
                      num_channels=3,
                      chunk_size=64,
                      rng=np.random,
                      prefix_path='train_raw/',
                      transfo_params=default_transfo_params,
                      paired_transfos=False):
    num_patients = len(images)
    indices = rng.randint(0, num_patients, chunk_size // 2)
    chunk_x = np.zeros((chunk_size, height, weight, num_channels), dtype='float32')
    chunk_y = np.zeros((chunk_size, 2), dtype='int32')

    for k, idx in enumerate(indices):
        # First eye.
        img = str(images[idx]) + '_left'
        chosen_values = load_image_and_process(img,
                                               im_dst=chunk_x[2 * k],
                                               output_shape=(height, weight),
                                               prefix_path=prefix_path,
                                               transfo_params=transfo_params)
        chunk_y[2 * k] = labels[idx][0]

        # Second eye.
        img = str(images[idx]) + '_right'
        load_image_and_process(img,
                               im_dst=chunk_x[2 * k + 1],
                               output_shape=(height, weight),
                               prefix_path=prefix_path,
                               transfo_params=transfo_params,
                               rand_values=chosen_values if paired_transfos else None)

        chunk_y[2 * k + 1] = labels[idx][1]
        yield chunk_x, np.eye(2)[chunk_y].astype('float32')


def update_labels(chunk_y, chunk_size=64):
    chunk_update = np.zeros((chunk_size, 2), dtype='int32')
    for i in range(chunk_size):
        if chunk_y[i][0] == 1:
            chunk_update[i][0] = 1
        else:
            chunk_update[i][1] = 1
    return chunk_update


def patches_gen_pairs_pseudolabel(images, labels,
                                  p_x=512, p_y=512,
                                  num_channels=3,
                                  num_chunk=1000,
                                  chunk_size=64,
                                  rng=np.random,
                                  train_path='train_raw/',
                                  test_path='test_raw/',
                                  transfo_params=default_transfo_params,
                                  paired_transfos=False):
    num_patients = len(images)
    for n in range(num_chunk):
        indices = rng.randint(0, num_patients, chunk_size//2)
        chunk_x = np.zeros((chunk_size, p_x, p_y, num_channels), dtype='float32')
        chunk_y = np.zeros((chunk_size, 5), dtype='float32')
        chunk_flag = np.zeros((chunk_size, 1), dtype='int32')
        int_labels = len(labels.shape) < 3
        id_matrix = np.eye(5)
        for k, idx in enumerate(indices):
            patient_id = images[idx]
            if patient_id in all_train_patient_ids:
                prefix_path = train_path
            else:
                prefix_path = test_path

            # First eye.
            img_id = str(patient_id) + '_left'
            chosen_values = load_image_and_process(img_id,
                                                   im_dst=chunk_x[2 * k],
                                                   flag = chunk_flag[2 * k],
                                                   output_shape=(p_x, p_y),
                                                   prefix_path=prefix_path,
                                                   transfo_params=transfo_params)

            if int_labels:
                chunk_y[2 * k] = id_matrix[int(labels[idx][0])]
            else:
                chunk_y[2 * k] = labels[idx][0]

            if chunk_flag[2 * k] == 1:
                if k > 0:
                    chunk_x[2 * k] = chunk_x[2 *(k - 1)]
                    chunk_y[2 * k] = chunk_y[2 *(k - 1)]

            # Second eye.
            img_id = str(patient_id) + '_right'
            load_image_and_process(img_id,
                                   im_dst=chunk_x[2 * k + 1],
                                   flag=chunk_flag[2 * k + 1],
                                   output_shape=(p_x, p_y),
                                   prefix_path=prefix_path,
                                   transfo_params=transfo_params,
                                   rand_values=chosen_values if paired_transfos else None)

            if int_labels:
                chunk_y[2 * k + 1] = id_matrix[int(labels[idx][1])]
            else:
                chunk_y[2 * k + 1] = labels[idx][1]

            if chunk_flag[2 * k + 1] == 1:
                if k > 0:
                    chunk_x[2 * k + 1] = chunk_x[2 *(k - 1) + 1]
                    chunk_y[2 * k + 1] = chunk_y[2 *(k - 1) + 1]

        chunk_update = update_labels(chunk_y, chunk_size) # update the label from 5 to 2
        yield chunk_x, chunk_update


def patch_data_pseudolabel(prefix_test):
    SEED = 1
    pl_enabled = True
    pl_softmax_temp = 2
    sample_coefs = [0, 7, 3, 22, 25]
    train_labels = p.read_csv('data/trainLabels.csv')
    labels_split = p.DataFrame(list(train_labels.image.str.split('_')), columns=['id', 'eye'])
    labels_split['level'] = train_labels.level
    labels_split['id'] = labels_split['id'].astype('int')

    id_train, y_train, id_valid, y_valid = split_data(train_labels,
                                                      labels_split,
                                                      valid_size=10,
                                                      SEED=SEED, pairs=True)
    if pl_enabled:
        pl_test_fn = '2017_09_08_log_mean.npy'
        test_preds = np.load('preds/' + pl_test_fn)
        if test_preds.shape[1] > 5:
            test_preds = test_preds[:, -5:].astype('float32')

        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)

        if np.mean(test_preds) > 0:
            test_preds = np.log(1e-5 + test_preds)

        test_probs = softmax(test_preds, temp=pl_softmax_temp)

        images_test_pl = sorted(set(get_img_ids_from_dir(prefix_test)))
        labels_test_pl = test_probs.reshape((-1, 2, 5))

        id_train_oversample, labels_train_oversample = oversample_set(id_train, y_train, sample_coefs)

        images_train = list(id_train_oversample) + images_test_pl
        labels_train_pl = np.eye(5)[
            list(labels_train_oversample.flatten().astype('int32'))
        ].reshape((-1, 2, 5))

        labels_train = np.vstack([labels_train_pl, labels_test_pl]).astype('float32')
    else:
        id_train_oversample, labels_train_oversample = oversample_set(id_train,
                                                                      y_train,
                                                                      sample_coefs)
        images_train = id_train_oversample
        labels_train = labels_train_oversample.astype('int32')

    return images_train, labels_train


def update_data(train_data, chunk_size):
    size = len(train_data)
    num = size // chunk_size
    return train_data[:num*chunk_size]


def generate_data(prefix_train, prefix_test,
                  image_width=512, image_height=512,
                  num_channels=3,
                  num_chunk=1000,
                  rng=np.random,
                  chunk_size=batch_size):
    images_train, labels_train = patch_data_pseudolabel(prefix_test)
    generator = patches_gen_pairs_pseudolabel(images=images_train,
                                              labels=labels_train,
                                              p_x=image_height, p_y=image_width,
                                              num_channels=num_channels,
                                              num_chunk=num_chunk,
                                              rng=rng,
                                              chunk_size=chunk_size,
                                              train_path=prefix_train,
                                              test_path=prefix_test,
                                              transfo_params=default_transfo_params,
                                              paired_transfos=True)
    return generator


def patches_gen_pairs_samples(pos_dir, neg_dir,
                              width=512, height=512,
                              num_channels=3,
                              num_chunk=1000,
                              chunk_size=64,
                              rng=np.random,
                              transfo_params=default_transfo_params):
    images_pos = os.listdir(pos_dir)
    num_pos = len(images_pos)
    images_neg = os.listdir(neg_dir)
    num_neg = len(images_neg)

    for n in range(num_chunk):
        indices_pos = rng.randint(0, num_pos, chunk_size // 2)
        indices_neg = rng.randint(0, num_neg, chunk_size // 2)
        chunk_x = np.zeros((chunk_size, width, height, num_channels), dtype='float32')
        chunk_y = np.zeros((chunk_size, 2), dtype='float32')
        chunk_flag = np.zeros((chunk_size, 1), dtype='int32')

        for k, idx in enumerate(indices_pos):
            image_name = images_pos[idx]
            load_image_and_process(image_name,
                                   im_dst=chunk_x[2 * k],
                                   flag = chunk_flag[2 * k],
                                   output_shape=(width, height),
                                   prefix_path=pos_dir,
                                   transfo_params=transfo_params)
            chunk_y[2 * k][0] = 1
            if chunk_flag[2 * k] == 1:
                if k > 0:
                    chunk_x[2 * k] = chunk_x[2 *(k - 1)]
                    chunk_y[2 * k] = chunk_y[2 *(k - 1)]

        for k, idx in enumerate(indices_neg):
            image_name = images_neg[idx]
            load_image_and_process(image_name,
                                   im_dst=chunk_x[2 * k + 1],
                                   flag=chunk_flag[2 * k + 1],
                                   output_shape=(width, height),
                                   prefix_path=neg_dir,
                                   transfo_params=transfo_params)
            chunk_y[2 * k + 1][1] = 1
            if chunk_flag[2 * k + 1] == 1:
                if k > 0:
                    chunk_x[2 * k + 1] = chunk_x[2 * (k - 1) + 1]
                    chunk_y[2 * k + 1] = chunk_y[2 * (k - 1) + 1]

        yield [chunk_x, chunk_x], chunk_y
        #yield chunk_x, chunk_y


def patches_gen_pairs_samples(pos_dir, neg_dir,
                              width=512, height=512,
                              num_channels=3,
                              num_chunk=1000,
                              chunk_size=64,
                              rng=np.random,
                              transfo_params=default_transfo_params):
    images_pos = os.listdir(pos_dir)
    num_pos = len(images_pos)
    images_neg = os.listdir(neg_dir)
    num_neg = len(images_neg)

    data = []
    label = []
    for n in range(num_chunk):
        indices_pos = rng.randint(0, num_pos, chunk_size // 2)
        indices_neg = rng.randint(0, num_neg, chunk_size // 2)
        chunk_x = np.zeros((chunk_size, width, height, num_channels), dtype='float32')
        chunk_y = np.zeros((chunk_size, 2), dtype='float32')
        chunk_flag = np.zeros((chunk_size, 1), dtype='int32')

        for k, idx in enumerate(indices_pos):
            image_name = images_pos[idx]
            load_image_and_process(image_name,
                                   im_dst=chunk_x[2 * k],
                                   flag = chunk_flag[2 * k],
                                   output_shape=(width, height),
                                   prefix_path=pos_dir,
                                   transfo_params=transfo_params)
            chunk_y[2 * k][0] = 1
            if chunk_flag[2 * k] == 1:
                if k > 0:
                    chunk_x[2 * k] = chunk_x[2 *(k - 1)]
                    chunk_y[2 * k] = chunk_y[2 *(k - 1)]
            data.append(chunk_x[2 * k])
            label.append(chunk_y[2 * k])
        for k, idx in enumerate(indices_neg):
            image_name = images_neg[idx]
            load_image_and_process(image_name,
                                   im_dst=chunk_x[2 * k + 1],
                                   flag=chunk_flag[2 * k + 1],
                                   output_shape=(width, height),
                                   prefix_path=neg_dir,
                                   transfo_params=transfo_params)
            chunk_y[2 * k + 1][1] = 1
            if chunk_flag[2 * k + 1] == 1:
                if k > 0:
                    chunk_x[2 * k + 1] = chunk_x[2 * (k - 1) + 1]
                    chunk_y[2 * k + 1] = chunk_y[2 * (k - 1) + 1]

            data.append(chunk_x[2 * k + 1])
            label.append(chunk_y[2 * k] + 1)
        #yield [chunk_x, chunk_x], chunk_y
        #yield chunk_x, chunk_y

    return np.array(data), np.array(label)


def patches_gen_pairs_validate(pos_dir, neg_dir,
                               width=512, height=512,
                               num_channels=3,
                               num_chunk=1000,
                               chunk_size=64,
                               rng=np.random,
                               transfo_params=default_transfo_params):
    images_pos = os.listdir(pos_dir)
    num_pos = len(images_pos)
    images_neg = os.listdir(neg_dir)
    num_neg = len(images_neg)

    data = []
    label = []
    for n in range(num_chunk):
        indices_pos = rng.randint(0, num_pos, chunk_size // 2)
        indices_neg = rng.randint(0, num_neg, chunk_size // 2)
        chunk_x = np.zeros((chunk_size, width, height, num_channels), dtype='float32')
        chunk_y = np.zeros((chunk_size, 2), dtype='float32')
        chunk_flag = np.zeros((chunk_size, 1), dtype='int32')

        for k, idx in enumerate(indices_pos):
            image_name = images_pos[idx]
            load_image_and_process(image_name,
                                   im_dst=chunk_x[2 * k],
                                   flag = chunk_flag[2 * k],
                                   output_shape=(width, height),
                                   prefix_path=pos_dir,
                                   transfo_params=transfo_params)
            chunk_y[2 * k][0] = 1
            if chunk_flag[2 * k] == 1:
                if k > 0:
                    chunk_x[2 * k] = chunk_x[2 *(k - 1)]
                    chunk_y[2 * k] = chunk_y[2 *(k - 1)]
            data.append(chunk_x[2 * k])
            label.append(chunk_y[2 * k])
        for k, idx in enumerate(indices_neg):
            image_name = images_neg[idx]
            load_image_and_process(image_name,
                                   im_dst=chunk_x[2 * k + 1],
                                   flag=chunk_flag[2 * k + 1],
                                   output_shape=(width, height),
                                   prefix_path=neg_dir,
                                   transfo_params=transfo_params)
            chunk_y[2 * k + 1][1] = 1
            if chunk_flag[2 * k + 1] == 1:
                if k > 0:
                    chunk_x[2 * k + 1] = chunk_x[2 * (k - 1) + 1]
                    chunk_y[2 * k + 1] = chunk_y[2 * (k - 1) + 1]

            data.append(chunk_x[2 * k + 1])
            label.append(chunk_y[2 * k] + 1)
        #yield [chunk_x, chunk_x], chunk_y
        #yield chunk_x, chunk_y

    return np.array(data), np.array(label)


image_name = '17_left.jpeg'
im_dst = np.zeros((512, 512, 3), dtype='float32')
load_image_and_process(image_name, im_dst,0,
                           output_shape=(512, 512),
                           prefix_path='data/',
                           transfo_params=default_transfo_params,
                           rand_values=None)

