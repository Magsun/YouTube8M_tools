import os, sys, re, subprocess as sb

import tensorflow as tf
import glob
import numpy


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _make_bytes(int_array):
    if bytes == str:  # Python2
        return ''.join(map(chr, int_array))
    else:
        return bytes(int_array)


def quantize(features, min_quantized_value=-2.0, max_quantized_value=2.0):
    """Quantizes float32 `features` into string."""
    assert features.dtype == 'float32'
    assert len(features.shape) == 1  # 1-D array
    features = numpy.clip(features, min_quantized_value, max_quantized_value)
    quantize_range = max_quantized_value - min_quantized_value
    features = (features - min_quantized_value) * (255.0 / quantize_range)
    features = [int(round(f)) for f in features]

    return _make_bytes(features)


def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    """
    Dequantize the feature from the byte format to the float format.

    Args:
    feat_vector: the input 1-d vector.
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.

    Returns:
    A float vector which has the same shape as feat_vector.
    """
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias


contexts = {
    'AUDIO/feature/dimensions': tf.io.FixedLenFeature([], tf.int64),
    'AUDIO/feature/rate': tf.io.FixedLenFeature([], tf.float32),
    'RGB/feature/dimensions': tf.io.FixedLenFeature([], tf.int64),
    'RGB/feature/rate': tf.io.FixedLenFeature([], tf.float32),
    'clip/data_path': tf.io.FixedLenFeature([], tf.string),
    'clip/end/timestamp': tf.io.FixedLenFeature([], tf.int64),
    'clip/start/timestamp': tf.io.FixedLenFeature([], tf.int64)
}

features = {
    'AUDIO/feature/floats': tf.io.VarLenFeature(dtype=tf.float32),
    'AUDIO/feature/timestamp': tf.io.VarLenFeature(tf.int64),
    'RGB/feature/floats': tf.io.VarLenFeature(dtype=tf.float32),
    'RGB/feature/timestamp': tf.io.VarLenFeature(tf.int64)

}


def parse_exmp(serial_exmp):
    _, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serial_exmp,
        context_features=contexts,
        sequence_features=features)

    sequence_parsed = tf.contrib.learn.run_n(sequence_parsed)[0]

    # audio : [10, 128]
    # rgb : [10, 1024]

    audio = sequence_parsed['AUDIO/feature/floats'].values
    rgb = sequence_parsed['RGB/feature/floats'].values

    # print(audio.values)
    # print(type(audio.values))

    audio_slices = [audio[128 * i: 128 * (i + 1)] for i in range(len(audio) // 128)]
    rgb_slices = [rgb[1024 * i: 1024 * (i + 1)] for i in range(len(rgb) // 1024)]

    byte_audio = []
    byte_rgb = []

    for seg in audio_slices:
        audio_seg = quantize(seg)
        byte_audio.append(audio_seg)

    for seg in rgb_slices:
        rgb_seg = quantize(seg)
        byte_rgb.append(rgb_seg)

    return byte_audio, byte_rgb


def make_exmp(id, labels, audio, rgb):
    audio_features = []
    rgb_features = []

    for embedding in audio:
        embedding_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[embedding]))
        audio_features.append(embedding_feature)

    for embedding in rgb:
        embedding_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[embedding]))
        rgb_features.append(embedding_feature)

    seq_exmp = tf.train.SequenceExample(
        context=tf.train.Features(
            feature={
                'id': tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[id.encode('utf-8')])),
                'labels': tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[labels]))
            }),
        feature_lists=tf.train.FeatureLists(
            feature_list={
                'audio': tf.train.FeatureList(
                    feature=audio_features
                ),
                'rgb': tf.train.FeatureList(
                    feature=rgb_features
                )
            })
    )
    serialized = seq_exmp.SerializeToString()
    return serialized


# data_root = '/data/rain_video/src_video'
data_root = '/data/thunder_video/ucf101/thunder/'

# for file_name in glob.glob(data_root + '/rain*.mp4'):
for file_name in glob.glob(data_root + '/thunder_*.mp4'):
    # print(file_name)

    step_1 = 'python -m mediapipe.examples.desktop.youtube8m.generate_input_sequence_example \
            --path_to_input_video=' + file_name + '\
            --clip_end_time_sec=10'

    # print(step_1)

    (status, output) = sb.getstatusoutput(step_1)  # 获得shell命令执行后的状态status和控制台的所有输出output
    # status：表示执行程序结果状态，值是0表示执行成功。
    # output：就是打印到控制台一个以\n为拼接的字符串。
    # print(status)

    base_name = file_name.split('/')[-1]
    base_name = base_name.split('.')[0]
    # print(base_name)

    if status == 0:
        step_2 = 'GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/youtube8m/extract_yt8m_features \
                --calculator_graph_config_file=mediapipe/graphs/youtube8m/feature_extraction.pbtxt \
                --input_side_packets=input_sequence_example=/tmp/mediapipe/metadata.pb  \
                --output_side_packets=output_sequence_example=/data/thunder_video/tmp_pb/' + base_name + '.pb'
        (status, output) = sb.getstatusoutput(step_2)  # 获得shell命令执行后的状态status和控制台的所有输出output
        print(output)

        filename = '/data/thunder_video/tmp_pb/' + base_name + '.pb'

        dst_dir = '/data/thunder_video/tfrecord/thunder/'

        sequence_example = open(filename, 'rb').read()

        audio, rgb = parse_exmp(sequence_example)

        id = base_name

        labels = 0  # positive=0 negative=1

        tmp_example = make_exmp(id, labels, audio, rgb)

        decoded = tf.train.SequenceExample.FromString(tmp_example)
        print(decoded)

        writer = tf.python_io.TFRecordWriter(dst_dir + base_name + '.tfrecord')
        writer.write(tmp_example)
