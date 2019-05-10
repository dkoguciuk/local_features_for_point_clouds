import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = os.path.basename(www)
    os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % zipfile)


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def _rotate(batch_data, rotation_angle=None):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        if rotation_angle is None:
            rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    if batch_data.shape[-1] == 3:
        return _rotate(batch_data)
    elif batch_data.shape[-1] == 6:
        coords = _rotate(batch_data[:, :, :3])
        normls = _rotate(batch_data[:, :, 3:])
        return np.concatenate((coords, normls), axis=-1)
    # Assert
    assert False, 'Wrong data size!'


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    if batch_data.shape[-1] == 3:
        return _rotate(batch_data, rotation_angle)
    elif batch_data.shape[-1] == 6:
        coords = _rotate(batch_data[:, :, :3], rotation_angle)
        normls = _rotate(batch_data[:, :, 3:], rotation_angle)
        return np.concatenate((coords, normls), axis=-1)
    # Assert
    assert False, 'Wrong data size!'


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    b, n, c = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(b, n, c), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


def get_data_files(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename, with_normals=False):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    labels = f['label'][:]
    if with_normals:
        normals = f['normal'][:]
        data = np.concatenate((data, normals), axis=-1)
    return data, labels


def load_data_file(filename, with_normals=False):
    return load_h5(filename, with_normals)


def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return data, label, seg


def load_data_file_with_seg(filename):
    return load_h5_data_label_seg(filename)
