
import torch
import os
import numpy as np
import os.path as osp
import datetime

from functools import partial
from matplotlib import colors
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from datasets.display import get_cmap
import matplotlib.cm as cm
from matplotlib.colors import Normalize

HMF_COLORS = np.array([
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219]
]) / 255

def gray2color_ir107(image, **kwargs):
    cmap, norm, vmin, vmax = get_cmap('ir107', encoded=True)
    cmap = cm.get_cmap(cmap)
    if norm is None:
        norm = Normalize(vmin=vmin, vmax=vmax)

    normalized_image = norm(image)
    colored_image = cmap(normalized_image)
    return colored_image


def pad_to_width(image, target_width, color=(0, 0, 0, 0)):
    """
    Pad a (H, W, 4) image to (H, target_width, 4) with the given RGBA color.
    """
    H, W, C = image.shape
    assert C == 4, "Image must have 4 channels (RGBA)"
    assert target_width >= W, "Target width must be >= current width"

    pad_width = target_width - W
    pad_right = pad_width

    # Create padding arrays
    right_pad = np.full((H, pad_right, 4), color, dtype=image.dtype)

    padded = np.concatenate([image, right_pad], axis=1)  # pad along width
    return padded

def insert_horizontal_white_lines(image, row_height=128, line_color=(1, 1, 1, 1)):
    """
    Inserts 1-pixel horizontal white lines between rows stacked vertically.
    Args:
        image: (H, W, 4) numpy array
        row_height: height of one logical row (default 128)
        line_color: RGBA color tuple for the line
    Returns:
        New image with separator lines inserted
    """
    H, W, C = image.shape
    assert C == 4, "Expected RGBA image"

    num_rows = H // row_height
    line = np.full((1, W, 4), line_color, dtype=image.dtype)
    rows_with_lines = []

    for i in range(num_rows):
        row_start = i * row_height
        row = image[row_start : row_start + row_height]
        rows_with_lines.append(row)
        if i < num_rows - 1:
            rows_with_lines.append(line)  # insert white line between rows

    return np.concatenate(rows_with_lines, axis=0)

def insert_vertical_white_lines(image, col_width=128, line_color=(1, 1, 1, 1)):
    """
    Inserts 1-pixel vertical white lines between image columns.
    Args:
        image: (H, W, 4) numpy array
        col_width: width of one logical column (e.g., 640)
        line_color: RGBA color tuple for the vertical line
    Returns:
        New image with vertical lines inserted
    """
    H, W, C = image.shape
    assert C == 4, "Expected RGBA image"

    num_cols = W // col_width
    line = np.full((H, 1, 4), line_color, dtype=image.dtype)
    cols_with_lines = []

    for i in range(num_cols):
        col_start = i * col_width
        col = image[:, col_start : col_start + col_width]
        cols_with_lines.append(col)
        if i < num_cols - 1:
            cols_with_lines.append(line)  # insert vertical line between columns

    return np.concatenate(cols_with_lines, axis=1)


def vis_res(input_seq, gt_seq, pred_seq, input_ir107, gt_ir107, save_path, data_type='vil',
            save_grays=False, do_hmf=False, save_colored=False,save_gif=False,
            pixel_scale = None, thresholds = None, gray2color = None
            ):
    # pred_seq: ndarray, [T, C, H, W], value range: [0, 1] float
    if isinstance(pred_seq, torch.Tensor) or isinstance(gt_seq, torch.Tensor):
        input_seq = input_seq.detach().cpu().numpy()
        pred_seq = pred_seq.detach().cpu().numpy()
        gt_seq = gt_seq.detach().cpu().numpy()
        input_ir107 = input_ir107.detach().cpu().numpy()
        gt_ir107 = gt_ir107.detach().cpu().numpy()

    pred_seq = pred_seq.squeeze()
    input_seq = input_seq.squeeze()
    gt_seq = gt_seq.squeeze()
    gt_ir107 = gt_ir107.squeeze()
    input_ir107 = input_ir107.squeeze()
    # os.makedirs(save_path, exist_ok=True)


    if data_type=='vil':
        pred_seq = pred_seq * pixel_scale
        pred_seq = pred_seq.astype(np.uint8)
        gt_seq = gt_seq * pixel_scale
        gt_seq = gt_seq.astype(np.uint8)
        input_seq = input_seq * pixel_scale
        input_seq = input_seq.astype(np.uint8)

        max_val = 2000
        min_val = -7000
        gt_ir107 = gt_ir107 * (max_val - min_val) + min_val
        input_ir107 = input_ir107 * (max_val - min_val) + min_val
    
    colored_input = np.array([gray2color(input_seq[i], data_type=data_type) for i in range(len(input_seq))], dtype=np.float64)
    colored_pred = np.array([gray2color(pred_seq[i], data_type=data_type) for i in range(len(pred_seq))], dtype=np.float64)
    colored_gt =  np.array([gray2color(gt_seq[i], data_type=data_type) for i in range(len(gt_seq))],dtype=np.float64)
    colored_ir107_gt =  np.array([gray2color_ir107(gt_ir107[i], data_type=data_type) for i in range(len(gt_ir107))],dtype=np.float64)
    colored_ir107_input =  np.array([gray2color_ir107(input_ir107[i], data_type=data_type) for i in range(len(input_ir107))],dtype=np.float64)


    grid_input = np.concatenate([
        np.concatenate([i for i in colored_input], axis=-2),
    ], axis=-3)
    grid_pred = np.concatenate([
        np.concatenate([i for i in colored_pred], axis=-2),
    ], axis=-3)
    grid_gt = np.concatenate([
        np.concatenate([i for i in colored_gt], axis=-2,),
    ], axis=-3)
    grid_ir107_gt = np.concatenate([
        np.concatenate([i for i in colored_ir107_gt], axis=-2,),
    ], axis=-3)
    grid_ir107_input = np.concatenate([
        np.concatenate([i for i in colored_ir107_input], axis=-2,),
    ], axis=-3)

    grid_input = pad_to_width(grid_input, target_width=grid_gt.shape[1])
    grid_ir107_input = pad_to_width(grid_ir107_input, target_width=grid_pred.shape[1])
    
    grid_concat = np.concatenate([grid_input, grid_gt, grid_pred, grid_ir107_input, grid_ir107_gt], axis=-3,)
    grid_concat = insert_horizontal_white_lines(grid_concat)
    grid_concat = insert_vertical_white_lines(grid_concat)
    plt.imsave(osp.join(save_path+'_all.png'), grid_concat)
    
    if save_gif:
        clip = ImageSequenceClip(list(colored_pred * 255), fps=4)
        clip.write_gif(osp.join(save_path, 'pred.gif'), fps=4, verbose=False)
        clip = ImageSequenceClip(list(colored_gt * 255), fps=4)
        clip.write_gif(osp.join(save_path, 'targets.gif'), fps=4, verbose=False)
    
    if do_hmf:
        def hit_miss_fa(y_true, y_pred, thres):
            mask = np.zeros_like(y_true)
            mask[np.logical_and(y_true >= thres, y_pred >= thres)] = 4
            mask[np.logical_and(y_true >= thres, y_pred < thres)] = 3
            mask[np.logical_and(y_true < thres, y_pred >= thres)] = 2
            mask[np.logical_and(y_true < thres, y_pred < thres)] = 1
            return mask
            
        grid_pred = np.concatenate([
            np.concatenate([i for i in pred_seq], axis=-1),
        ], axis=-2)
        grid_gt = np.concatenate([
            np.concatenate([i for i in gt_seq], axis=-1),
        ], axis=-2)

        hmf_mask = hit_miss_fa(grid_pred, grid_gt, thres=thresholds[2])
        plt.axis('off')
        plt.imsave(osp.join(save_path, 'hmf.png'), hmf_mask, cmap=colors.ListedColormap(HMF_COLORS))

DATAPATH = {
    'meteo' : 'MeteoNet.h5',
    'sevir' : '/home/pwen5103/Weather/DiffCastB/data/sevir'
}

def get_dataset(data_name, img_size, seq_len, **kwargs):
    dataset_name = data_name.lower()
    train = val = test = None
    
    if data_name == 'meteo':
        from .dataset_meteonet import Meteo, gray2color, THRESHOLDS, PIXEL_SCALE
        train = Meteo(DATAPATH[data_name], type='train', img_size=img_size)
        val = Meteo(DATAPATH[data_name], type='test', img_size=img_size)
        test = Meteo(DATAPATH[data_name], type='test', img_size=img_size)
        
    elif dataset_name == 'sevir':
        from .dataset_sevir import SEVIRTorchDataset, gray2color, PIXEL_SCALE, THRESHOLDS
        
        train_valid_split = (2019, 1, 1)
        valid_test_split = (2019, 6, 1)
        batch_size = kwargs.get('batch_size', 1)

        stride = 13
        
        train = SEVIRTorchDataset(
            dataset_dir=DATAPATH[data_name],
            split_mode='uneven',
            img_size=img_size,
            shuffle=True,
            seq_len=seq_len,
            stride=stride,      # ?
            sample_mode='sequent',
            batch_size=batch_size,
            num_shard=1,
            rank=0,
            start_date=None,
            end_date=datetime.datetime(*train_valid_split),
            output_type=np.float32,
            preprocess=True,
            rescale_method='01',
            verbose=False
        )
        
        val = SEVIRTorchDataset(
            dataset_dir=DATAPATH[data_name],
            split_mode='uneven',
            img_size=img_size,
            shuffle=False,
            seq_len=seq_len,
            stride=stride,      # ?
            sample_mode='sequent',
            batch_size=batch_size * 2,
            num_shard=1,
            rank=0,
            start_date=datetime.datetime(*train_valid_split),
            end_date=datetime.datetime(*valid_test_split),
            output_type=np.float32,
            preprocess=True,
            rescale_method='01',
            verbose=False
        )
        
        test = SEVIRTorchDataset(
            dataset_dir=DATAPATH[data_name],
            split_mode='uneven',
            shuffle=False,
            img_size=img_size,
            seq_len=seq_len,
            stride=stride,      # ?
            sample_mode='sequent',
            batch_size=batch_size * 2,
            num_shard=1,
            rank=0,
            start_date=datetime.datetime(*valid_test_split),
            end_date=None,
            output_type=np.float32,
            preprocess=True,
            rescale_method='01',
            verbose=False
        )
        

    color_fn = partial(vis_res, 
                    pixel_scale = PIXEL_SCALE, 
                    thresholds = THRESHOLDS, 
                    gray2color = gray2color)
    
    return train, val, test, color_fn, PIXEL_SCALE, THRESHOLDS
