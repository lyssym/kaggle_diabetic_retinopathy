# coding: utf-8

batch_size = 4
num_chunk = 10
height = 512
width = 512

default_transfo_params = {'rotation': True, 'rotation_range': (0, 360),
                          'contrast': True, 'contrast_range': (0.7, 1.3),
                          'brightness': True, 'brightness_range': (0.7, 1.3),
                          'color': True, 'color_range': (0.7, 1.3),
                          'flip': True, 'flip_prob': 0.5,
                          'crop': True, 'crop_prob': 0.4,
                          'crop_w': 0.03, 'crop_h': 0.04,
                          'keep_aspect_ratio': False,
                          'resize_pad': False,
                          'zoom': True, 'zoom_prob': 0.5,
                          'zoom_range': (0.00, 0.05),
                          'paired_transfos': False,
                          'rotation_expand': False,
                          'crop_height': False,
                          'extra_width_crop': True,
                          'rotation_before_resize': False,
                          'crop_after_rotation': True}

no_transfo_params = {'keep_aspect_ratio': default_transfo_params['keep_aspect_ratio'],
                     'resize_pad': default_transfo_params['resize_pad'],
                     'extra_width_crop': default_transfo_params['extra_width_crop'],
                     'rotation_before_resize': default_transfo_params['rotation_before_resize'],
                     'crop_height': default_transfo_params['crop_height'],
                     }