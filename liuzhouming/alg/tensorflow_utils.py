import tensorflow as tf


def try_gpu(i=0):  # @save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if len(tf.config.experimental.list_physical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')


def try_all_gpus():  # @save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    devices = [tf.device(f'/GPU:{i}') for i in range(num_gpus)]
    return devices if devices else [tf.device('/CPU:0')]


def try_device(device):
    device = device.lower()
    if 'cuda' in device:
        device_split = device.split(':')
        return try_gpu(int(device_split[-1]))
    else:
        return tf.device('/CPU:0')
