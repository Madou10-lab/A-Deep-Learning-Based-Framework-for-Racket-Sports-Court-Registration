import pandas as pd
import label_definition as ld
import os.path as osp
import os
from sklearn.model_selection import train_test_split
import numpy as np
import utils
import dataset_utils as du
import albumentations as album
import logging
import ast
from PIL import Image, ImageDraw
import cv2
import shutil

logger = logging.getLogger(__name__)


class TennisDataset:
    def __init__(self,
                 experiment_id,
                 dataset_name,
                 labels_path,
                 dataset_path,
                 experiment_path,
                 input_height,
                 input_width,
                 shuffle,
                 augmentation_colour_format,
                 augmentation_spatial,
                 augmentation_colour,
                 split_ratio
                 ):
        self.experiment_id = experiment_id
        self.dataset_name = dataset_name
        self.input_height = input_height
        self.input_width = input_width
        self.shuffle = shuffle
        self.augmentation_colour_format = augmentation_colour_format
        self.augmentation_spatial = augmentation_spatial
        self.augmentation_colour = augmentation_colour
        self.split_ratio = split_ratio
        self.labels_path = labels_path
        self.dataset_path = dataset_path
        self.experiment_path = experiment_path

        self.logfilehandler = logging.FileHandler(osp.join(experiment_path, "logs", "dataset_output.log"), 'a')
        self.logfilehandler.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
        logger.addHandler(self.logfilehandler)
        logger.info(f"Experiment number {self.experiment_id}")
        logger.info(self.__class__.__name__ + " instance created")

    def prepare(self):
        self.prepare_dataset()
        self.get_class()
        #self.generate_labels()
        self.split_dataset()
        self.setup_augmentation()
        self.build_vis()

    def prepare_dataset(self):
        temp_path = osp.join(self.experiment_path, "Temp_dataset")

        train_dir = osp.join(temp_path, 'train')
        valid_dir = osp.join(temp_path, 'valid')
        test_dir = osp.join(temp_path, 'test')

        self.x_train_dir = osp.join(train_dir, 'source')
        self.y_train_dir = osp.join(train_dir, 'mask')

        self.x_valid_dir = osp.join(valid_dir, 'source')
        self.y_valid_dir = osp.join(valid_dir, 'mask')

        self.x_test_dir = osp.join(test_dir, 'source')
        self.y_test_dir = osp.join(test_dir, 'mask')

        utils.create_folder(temp_path)

        utils.create_folder(train_dir)
        utils.create_folder(valid_dir)
        utils.create_folder(test_dir)

        utils.create_folder(self.x_train_dir)
        utils.create_folder(self.y_train_dir)

        utils.create_folder(self.x_valid_dir)
        utils.create_folder(self.y_valid_dir)

        utils.create_folder(self.x_test_dir)
        utils.create_folder(self.y_test_dir)

        logger.info("Dataset temporary folders created")

    def get_class(self, class_names=None,back_names=None):
        if class_names is None:
            class_names = list(ld.mask_ids.keys())
        self.class_ids = [ld.mask_ids[x] for x in class_names]
        self.back_ids =[] if back_names is None else [class_names.index(x) for x in back_names]
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.colour_palette = [ld.colour_palette[i] for i in self.class_ids]


        logger.info(f"Class names: {self.class_names}")
        logger.info(f"Class ids: {self.class_ids}")
        logger.info(f"Class colour palette: {self.colour_palette}")

    def generate_labels(self):
        mask_dir_path = osp.join(self.labels_path, "Mask", self.dataset_name+"_"+str(self.input_height))
        if osp.exists(mask_dir_path):
            logger.info("Dataset labels already generated")
            return
        os.makedirs(mask_dir_path)
        mask_data_file = osp.join(self.labels_path, 'mask_dataset_labels.csv')
        mask_df = pd.read_csv(mask_data_file)
        for index in mask_df.index:
            mask = Image.new('L', (self.input_width, self.input_height), 0)
            if pd.isna(mask_df["full_court"][index]):
                mask.save(osp.join(mask_dir_path, f"{mask_df['image_id'][index]}.png"))
                continue
            for i, cls in enumerate(self.class_names):
                if cls == 'background':
                    continue
                points = ast.literal_eval(mask_df[cls][index])
                polygon = [(round(self.input_width * points[i] / 100, 2), round(self.input_height * points[i + 1] / 100, 2)) for i in
                           range(0, len(points), 2)]
                ImageDraw.Draw(mask).polygon(polygon, fill=i)
            mask.save(osp.join(mask_dir_path, f"{mask_df['image_id'][index]}.png"))
        logger.info("Dataset labels generated")


    def split_dataset(self):
        metadata_file = osp.join(self.labels_path, 'metadata_dataset_labels.csv')
        source_path = osp.join(self.dataset_path, "Source")
        mask_path = osp.join(self.labels_path, "Mask", self.dataset_name+"_"+str(self.input_height))
        metadata_df = pd.read_csv(metadata_file)

        # Split the data into training and validation sets
        train_ids, valid_ids = train_test_split(metadata_df['image_id'], train_size=self.split_ratio, shuffle=self.shuffle)

        # Loop through training data and copy images and masks
        for image_id in train_ids:
            shutil.copy(os.path.join(source_path, image_id), os.path.join(self.x_train_dir, image_id))
            shutil.copy(os.path.join(mask_path, image_id), os.path.join(self.y_train_dir, image_id))

        # Loop through validation data and copy images and masks
        for image_id in valid_ids:
            shutil.copy(os.path.join(source_path, image_id), os.path.join(self.x_valid_dir, image_id))
            shutil.copy(os.path.join(mask_path, image_id), os.path.join(self.y_valid_dir, image_id))

        logger.info("Dataset splitted successfully into train/valid sets")

    def setup_augmentation(self):
        train_transform = [
            album.Resize(height=self.input_height, width=self.input_width, always_apply=True)
        ]
        if self.augmentation_colour_format == "gray":
            train_transform.append(album.ToGray(always_apply=True))
        if self.augmentation_colour_format == "hsv":
            train_transform.append(du.ToHSV(always_apply=True))
        if self.augmentation_colour_format == "bgr":
            train_transform.append(du.ToBGR(always_apply=True))
        if self.augmentation_spatial:
            train_transform.append(album.Perspective(scale=(0.05), always_apply=False, p=0.6))
            train_transform.append(album.ShiftScaleRotate(rotate_limit=(-3, 3), scale_limit=(0.05),
                                                          border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0,
                                                          shift_limit_x=(0, 0), shift_limit_y=(0, 0),
                                                          always_apply=False, p=0.6))
        if self.augmentation_colour:
            train_transform.append(album.ColorJitter(hue=0, always_apply=False, p=0.5))
            train_transform.append(album.GaussNoise(var_limit=10, p=0.1))

        test_transform = [
            album.Resize(height=self.input_height, width=self.input_width, always_apply=True)
        ]
        if self.augmentation_colour_format == "gray":
            test_transform.append(album.ToGray(always_apply=True))
        if self.augmentation_colour_format == "hsv":
            test_transform.append(du.ToHSV(always_apply=True))
        if self.augmentation_colour_format == "bgr":
            test_transform.append(du.ToBGR(always_apply=True))

        self.train_augmentation = album.Compose(train_transform)
        self.test_augmentation = album.Compose(test_transform)

    def build_vis(self):
        self.train_dataset_vis = du.BuildingsDataset(
            self.x_train_dir, self.y_train_dir,
            augmentation=self.train_augmentation,
            n_classes=self.n_classes,
        )

        self.valid_dataset_vis = du.BuildingsDataset(
            self.x_valid_dir, self.y_valid_dir,
            augmentation=self.test_augmentation,
            n_classes=self.n_classes,
        )

        #self.test_dataset_vis = du.BuildingsDataset(
        #    self.x_test_dir, self.y_test_dir,
        #    augmentation=self.test_augmentation,
        #    n_classes=self.n_classes,
        #)
        logger.info("Visual datasets built")

    def build_train(self, preprocessing_fn):
        self.train_dataset = du.BuildingsDataset(
            self.x_train_dir, self.y_train_dir,
            augmentation=self.test_augmentation,
            preprocessing=preprocessing_fn,
            n_classes=self.n_classes,
        )

        self.valid_dataset = du.BuildingsDataset(
            self.x_valid_dir, self.y_valid_dir,
            augmentation=self.test_augmentation,
            preprocessing=preprocessing_fn,
            n_classes=self.n_classes,
        )

        #self.test_dataset = du.BuildingsDataset(
        #    self.x_test_dir, self.y_test_dir,
        #    augmentation=self.test_augmentation,
        #    preprocessing=preprocessing_fn,
        #    n_classes=self.n_classes,
        #)
        logger.info("Preprocessed datasets built")

    def size(self):
        return self.train_size() + self.valid_size() + self.test_size()

    def train_size(self):
        return len(os.listdir(self.x_train_dir))

    def valid_size(self):
        return len(os.listdir(self.x_valid_dir))

    def test_size(self):
        return len(os.listdir(self.x_test_dir))

    def __len__(self):
        return self.n_classes

    def get_results(self, config):
        config["n_classes"] = self.n_classes
        config["train_size"] = self.train_size()
        config["valid_size"] = self.valid_size()
        #config["test_size"] = self.test_size()

    def __del__(self):
        logger.removeHandler(self.logfilehandler)


class CourtzonesDataset(TennisDataset):
    # Modified
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_class(self, class_names=None, back_names=None):
        class_names = ['background','front_no_mans_land','left_doubles','ad_court','deuce_court','back_no_mans_land','right_doubles']
        super().get_class(class_names)

class CourtzoneswithnetDataset(TennisDataset):
    # Modified
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_class(self, class_names=None, back_names=None):
        class_names = ['background','front_no_mans_land','left_doubles','ad_court','deuce_court','back_no_mans_land','right_doubles','net']
        super().get_class(class_names)

class FullcourtDataset(TennisDataset):
    # Modified
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_class(self, class_names=None, back_names=None):
        class_names = ['background','full_court']
        super().get_class(class_names)

class FrontDataset(TennisDataset):
    # Modified
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_class(self, class_names=None, back_names=None):
        class_names = ['background','front_no_mans_land']
        super().get_class(class_names)

