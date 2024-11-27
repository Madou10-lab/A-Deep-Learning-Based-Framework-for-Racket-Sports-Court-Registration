import os.path as osp
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp
import os


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"
    memory_used_info = sp.check_output(command.split()).decode('ascii').split('\n')[1:-1]
    memory_used_value, memory_total_value = [int(x.split()[0]) for x in memory_used_info[0].split(',')]
    return memory_used_value


def create_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def save_from_vids(metadata_df, vid_ids, path_from, path_to):
    for index in metadata_df.index:
        image_id = metadata_df['image_id'][index]
        video_id = metadata_df.loc[(metadata_df['image_id'] == image_id)]['video_id'].iloc[0]

        if video_id in vid_ids:
            shutil.copy(osp.join(path_from, f"{image_id}.png"),
                        osp.join(path_to, f"{image_id}.png"))

def save_from_vids_exists(metadata_df, vid_ids, path_from, path_to):
    for index in metadata_df.index:
        image_id = metadata_df['image_id'][index]
        exist = metadata_df['exist'][index]
        video_id = metadata_df.loc[(metadata_df['image_id'] == image_id)]['video_id'].iloc[0]

        if video_id in vid_ids and exist=='Yes':
            shutil.copy(osp.join(path_from, f"{image_id}.png"),
                        osp.join(path_to, f"{image_id}.png"))

def save_from_vids_occlusion(metadata_df, vid_ids, path_from, path_to):
    for index in metadata_df.index:
        image_id = metadata_df['image_id'][index]
        occlusion = metadata_df['occlusion'][index]
        video_id = metadata_df.loc[(metadata_df['image_id'] == image_id)]['video_id'].iloc[0]

        if video_id in vid_ids and occlusion=='No':
            shutil.copy(osp.join(path_from, f"{image_id}.png"),
                        osp.join(path_to, f"{image_id}.png"))

def save_from_vids_doubles(metadata_df, vid_ids, path_from, path_to):
    for index in metadata_df.index:
        image_id = metadata_df['image_id'][index]
        players = int(metadata_df['players'][index])
        video_id = metadata_df.loc[(metadata_df['image_id'] == image_id)]['video_id'].iloc[0]

        if video_id in vid_ids and players<3:
            shutil.copy(osp.join(path_from, f"{image_id}.png"),
                        osp.join(path_to, f"{image_id}.png"))


def visualize(**images):
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

def reverse_one_hot(image):
    x = np.argmax(image, axis=-1)
    return x


def transpose_reverse_one_hot(image):
    return reverse_one_hot(np.transpose(image, (1, 2, 0)))

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x


def generate_overlay(mask, source, opacity, colour_palette):
    pred_mask_seg = colour_code_segmentation(mask, colour_palette)
    img = cv2.addWeighted(source.astype(np.uint8), 1 - opacity, pred_mask_seg.astype(np.uint8), opacity, 0)
    return img


def generate_mask_fp(pred_mask, gt_mask, source, opacity, colour_palette):
    colour_palette_f=colour_palette[:]
    # create a boolean mask indicating where the two masks differ
    diff_mask = (gt_mask == pred_mask)
    # create a copy of the ground truth mask and set the differing pixels to zero
    color_seg = np.copy(gt_mask)
    color_seg[diff_mask] = len(colour_palette_f)
    colour_palette_f[0] = [45, 12, 230]
    colour_palette_f.append([0, 0, 0])
    pred_mask_seg = colour_code_segmentation(color_seg, colour_palette_f)
    img = cv2.addWeighted(source.astype(np.uint8), 1 - opacity, pred_mask_seg.astype(np.uint8), opacity, 0)
    return img


def concat_images(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])




