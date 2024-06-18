
import sys
import os
import errno
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F

import torch

from torch.utils.data import Subset
import numpy as np
import math


class PolySGDOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1



class PolyOptimizer(torch.optim.AdamW):

    def __init__(self, params, lr, max_step):
        super().__init__(params, lr)

        self.global_step = 0
        self.max_step = max_step

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** 0.9

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1

def compute_miou(pred, target, nclass=2):
    mini = 1

    # 计算公共区域
    intersection = pred * (pred == target)

    # 直方图
    area_inter, _ = np.histogram(intersection, bins=2, range=(mini, nclass))
    area_pred, _ = np.histogram(pred, bins=2, range=(mini, nclass))
    area_target, _ = np.histogram(target, bins=2, range=(mini, nclass))
    area_union = area_pred + area_target - area_inter

    # 交集已经小于并集
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"

    rate = round(max(area_inter) / max(area_union), 4)
    return rate

def dice_coeff(SR, GT):
    smooth = 1.
    num = SR.size(0)
    m1 = SR.view(num, -1)  # Flatten
    m2 = GT.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
 
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

class Logger(object):
    """
        Write console output to external text file.
        Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
        """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()
        
    def __enter__(self):
        pass
        
    def __exit__(self, *args):
        self.close()
        
    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())
        
    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise




class PathIndex:

    def __init__(self, radius, default_size):
        self.radius = radius
        self.radius_floor = int(np.ceil(radius) - 1)

        self.search_paths, self.search_dst = self.get_search_paths_dst(self.radius)

        self.path_indices, self.src_indices, self.dst_indices = self.get_path_indices(default_size)

        return

    def get_search_paths_dst(self, max_radius=5):

        coord_indices_by_length = [[] for _ in range(max_radius * 4)]

        search_dirs = []

        for x in range(1, max_radius):
            search_dirs.append((0, x))

        for y in range(1, max_radius):
            for x in range(-max_radius + 1, max_radius):
                if x * x + y * y < max_radius ** 2:
                    search_dirs.append((y, x))

        for dir in search_dirs:

            length_sq = dir[0] ** 2 + dir[1] ** 2
            path_coords = []

            min_y, max_y = sorted((0, dir[0]))
            min_x, max_x = sorted((0, dir[1]))

            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):

                    dist_sq = (dir[0] * x - dir[1] * y) ** 2 / length_sq

                    if dist_sq < 1:
                        path_coords.append([y, x])

            path_coords.sort(key=lambda x: -abs(x[0]) - abs(x[1]))
            path_length = len(path_coords)

            coord_indices_by_length[path_length].append(path_coords)

        path_list_by_length = [np.asarray(v) for v in coord_indices_by_length if v]
        path_destinations = np.concatenate([p[:, 0] for p in path_list_by_length], axis=0)

        return path_list_by_length, path_destinations

    def get_path_indices(self, size):

        full_indices = np.reshape(np.arange(0, size[0] * size[1], dtype=np.int64), (size[0], size[1]))

        cropped_height = size[0] - self.radius_floor
        cropped_width = size[1] - 2 * self.radius_floor

        path_indices = []

        for paths in self.search_paths:

            path_indices_list = []
            for p in paths:

                coord_indices_list = []

                for dy, dx in p:
                    coord_indices = full_indices[dy:dy + cropped_height,
                                    self.radius_floor + dx:self.radius_floor + dx + cropped_width]
                    coord_indices = np.reshape(coord_indices, [-1])

                    coord_indices_list.append(coord_indices)

                path_indices_list.append(coord_indices_list)

            path_indices.append(np.array(path_indices_list))

        src_indices = np.reshape(full_indices[:cropped_height, self.radius_floor:self.radius_floor + cropped_width], -1)
        dst_indices = np.concatenate([p[:,0] for p in path_indices], axis=0)

        return path_indices, src_indices, dst_indices


def edge_to_affinity(edge, paths_indices):

    aff_list = []
    edge = edge.view(edge.size(0), -1)

    for i in range(len(paths_indices)):
        if isinstance(paths_indices[i], np.ndarray):
            paths_indices[i] = torch.from_numpy(paths_indices[i])
        paths_indices[i] = paths_indices[i].cuda(non_blocking=True)

    for ind in paths_indices:
        ind_flat = ind.view(-1)
        dist = torch.index_select(edge, dim=-1, index=ind_flat)
        dist = dist.view(dist.size(0), ind.size(0), ind.size(1), ind.size(2))
        aff = torch.squeeze(1 - F.max_pool2d(dist, (dist.size(2), 1)), dim=2)
        aff_list.append(aff)
    aff_cat = torch.cat(aff_list, dim=1)

    return aff_cat


def affinity_sparse2dense(affinity_sparse, ind_from, ind_to, n_vertices):

    ind_from = torch.from_numpy(ind_from)
    ind_to = torch.from_numpy(ind_to)

    affinity_sparse = affinity_sparse.view(-1).cpu()
    ind_from = ind_from.repeat(ind_to.size(0)).view(-1)
    ind_to = ind_to.view(-1)

    indices = torch.stack([ind_from, ind_to])
    indices_tp = torch.stack([ind_to, ind_from])

    indices_id = torch.stack([torch.arange(0, n_vertices).long(), torch.arange(0, n_vertices).long()])

    affinity_dense = torch.sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                       torch.cat([affinity_sparse, torch.ones([n_vertices]), affinity_sparse])).to_dense().cuda()

    return affinity_dense


def to_transition_matrix(affinity_dense, beta, times):
    scaled_affinity = torch.pow(affinity_dense, beta)

    trans_mat = scaled_affinity / torch.sum(scaled_affinity, dim=0, keepdim=True)
    for _ in range(times):
        trans_mat = torch.matmul(trans_mat, trans_mat)

    return trans_mat


def propagate_to_edge(x, edge, radius=5, beta=10, exp_times=8):

    height, width = x.shape[-2:]

    hor_padded = width+radius*2
    ver_padded = height+radius

    path_index = PathIndex(radius=radius, default_size=(ver_padded, hor_padded))

    edge_padded = F.pad(edge, (radius, radius, 0, radius), mode='constant', value=1.0)
    sparse_aff = edge_to_affinity(torch.unsqueeze(edge_padded, 0),
                                  path_index.path_indices)
    dense_aff = affinity_sparse2dense(sparse_aff, path_index.src_indices,
                                      path_index.dst_indices, ver_padded * hor_padded)
    dense_aff = dense_aff.view(ver_padded, hor_padded, ver_padded, hor_padded)
    dense_aff = dense_aff[:-radius, radius:-radius, :-radius, radius:-radius]
    dense_aff = dense_aff.reshape(height * width, height * width)

    trans_mat = to_transition_matrix(dense_aff, beta=beta, times=exp_times)

    x = x.view(-1, height, width) * (1 - edge)

    rw = torch.matmul(x.view(-1, height * width), trans_mat)
    rw = rw.view(rw.size(0), 1, height, width)

    return rw

####
def propagate_to_aff(x, aff, radius=5, beta=10, exp_times=8):

    height, width = x.shape[-2:]

    hor_padded = width+radius*2
    ver_padded = height+radius

    path_index = PathIndex(radius=radius, default_size=(ver_padded, hor_padded))

    sparse_aff = F.pad(aff, (radius, radius, 0, radius), mode='constant', value=1.0)

    sparse_aff = sparse_aff.view(1, sparse_aff.shape[0], -1)

    dense_aff = affinity_sparse2dense(sparse_aff, path_index.src_indices,
                                      path_index.dst_indices, ver_padded * hor_padded)
    dense_aff = dense_aff.view(ver_padded, hor_padded, ver_padded, hor_padded)
    dense_aff = dense_aff[:-radius, radius:-radius, :-radius, radius:-radius]
    dense_aff = dense_aff.reshape(height * width, height * width)

    trans_mat = to_transition_matrix(dense_aff, beta=beta, times=exp_times)

    x = x.view(-1, height, width)

    rw = torch.matmul(x.view(-1, height * width), trans_mat)
    rw = rw.view(rw.size(0), 1, height, width)

    return rw