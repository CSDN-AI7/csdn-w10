#!/usr/bin/env python3
import logging
import os

import cv2
import numpy as np
import tensorflow as tf
from vgg import vgg_16


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')

FLAGS = flags.FLAGS

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']
# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [
                128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]


cm2lbl = np.zeros(256**3)
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def image2label(im):
    data = im.astype('int32')
    # cv2.imread. default channel layout is BGR
    idx = (data[:, :, 2] * 256 + data[:, :, 1]) * 256 + data[:, :, 0]
    return np.array(cm2lbl[idx])

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))  


def dict_to_tf_example(data, label):
    #with open(data, 'rb') as inf:
    #    encoded_data = inf.read()
    image_data = tf.gfile.FastGFile(data, 'rb').read()    # 生成图片数据，且是byte类型
    
    img_label = cv2.imread(label)
    img_mask = image2label(img_label)
    encoded_label = img_mask.astype(np.uint8).tobytes()

    height, width = img_label.shape[0], img_label.shape[1]
    if height < vgg_16.default_image_size or width < vgg_16.default_image_size:
        # 保证最后随机裁剪的尺寸
        return None

    filename = data.strip().split('/')[-1]                # 取出文件名信息
    _format=filename.strip().split('.')[-1]               # 取出文件的类型信息

    # Your code here, fill the dict
    feature_dict = {
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename.encode('utf8')),
        'image/encoded': bytes_feature(image_data),     
        'image/label': bytes_feature(encoded_label),
        'image/format': bytes_feature(_format.encode('utf8')),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(output_filename, file_pars):
    writer = tf.python_io.TFRecordWriter(output_filename)
    file_pairs_list=list(file_pars)   # 将字典结构改为list结构
    for idx, example in enumerate(file_pairs_list):
        data = example[0]              # 取出image data
        label = example[1]             # 取出image label
        if idx % 200 == 0:
            print( 'On image %d of %d'%( idx, len(file_pairs_list) ) )  # 打印进度信息
        try:
            tf_example = dict_to_tf_example(data, label)                # 按照tf_example协议制作tf_example数据结构
            if (tf_example!=None):
                writer.write(tf_example.SerializeToString())            # 生成 TFRecord数据 
            else:
                print('Invalid example: {}. small image. ignoring.'.format(example))  # 忽略尺寸不满足VGG16要求的图片
        except ValueError:
            print('Invalid example: {}. ignoring.'.format(example))

    writer.close()


def read_images_names(root, train=True):
    txt_fname = os.path.join(root, 'ImageSets/Segmentation/', 'train.txt' if train else 'val.txt')

    with open(txt_fname, 'r') as f:
        images = f.read().split()

    data = []
    label = []
    for fname in images:
        data.append('%s/JPEGImages/%s.jpg' % (root, fname))
        label.append('%s/SegmentationClass/%s.png' % (root, fname))
    return zip(data, label)


def main(_):
    logging.info('Prepare dataset file names')

    train_output_path = os.path.join(FLAGS.output_dir, 'fcn_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'fcn_val.record')

    train_files = read_images_names(FLAGS.data_dir, True)
    val_files = read_images_names(FLAGS.data_dir, False)
    print('*********create tfrecord of train*********\n')
    create_tf_record(train_output_path, train_files)
    print('*********create tfrecord of val*********\n')
    create_tf_record(val_output_path, val_files)


if __name__ == '__main__':
    tf.app.run()
