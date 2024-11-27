import os
import os.path as osp
from zipfile import ZipFile


def compress_dataset(config):
    dataset_path = config["dataset_path"]
    labels_path = config["labels_path"]
    source_dir_path = osp.join(labels_path, 'Source')
    mask_dir_path = osp.join(labels_path, 'Mask')
    test_vids_path = config["test_vids_path"]
    raw_label_path = config["raw_label_path"]
    zip_file = osp.join(dataset_path, "badminton_dataset.zip")

    print('Zipping dataset')

    file_paths = []

    for directory in [source_dir_path, mask_dir_path, test_vids_path]:
        rootdir = os.path.basename(directory)
        for root, directories, files in os.walk(directory):
            for filename in files:
                filepath = os.path.join(root, filename)
                parentpath = os.path.relpath(filepath, directory)
                arcname = os.path.join(rootdir, parentpath)
                if (filepath, arcname) not in file_paths:
                    file_paths.append((filepath, arcname))

    file_paths.append((osp.join(labels_path, 'metadata_dataset_labels.csv'), 'metadata_dataset_labels.csv'))
    file_paths.append((osp.join(labels_path, 'mask_dataset_labels.csv'), 'mask_dataset_labels.csv'))
    file_paths.append((raw_label_path, 'raw_dataset_labels.json'))
    file_paths.append((osp.join(dataset_path, 'video_urls.txt'), 'video_urls.txt'))
    file_paths.append((osp.join(dataset_path, 'badminton_video_list.csv'), 'badminton_video_list.csv'))
    file_paths.append((osp.join(dataset_path, 'badminton_image_list.csv'), 'badminton_image_list.csv'))

    count = 0
    with ZipFile(zip_file, 'w') as zip:
        for file in file_paths:
            zip.write(file[0], arcname=file[1])
            count += 1
            print(f"\rFiles processed: {count}/{len(file_paths)}")

    print('Dataset zipped successfully!')


if __name__ == '__main__':
    from config import load_config

    config = load_config()
    compress_dataset(config)
