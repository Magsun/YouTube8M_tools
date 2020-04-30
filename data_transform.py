# tensorflow=1.14

import tensorflow as tf
import glob, os
import numpy


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


# for parse feature.pb

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

    audio = sequence_parsed['AUDIO/feature/floats'].values
    rgb = sequence_parsed['RGB/feature/floats'].values

    # print(audio.values)
    # print(type(audio.values))

    # audio is 128 8bit, rgb is 1024 8bit for every second
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

    # for construct yt8m data
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


if __name__ == '__main__':
    filename = 'feature.pb'

    sequence_example = open(filename, 'rb').read()

    audio, rgb = parse_exmp(sequence_example)

    id = 'test_001'

    labels = 1

    tmp_example = make_exmp(id, labels, audio, rgb)

    decoded = tf.train.SequenceExample.FromString(tmp_example)
    print(decoded)

    # then you can write tmp_example to tfrecord files
