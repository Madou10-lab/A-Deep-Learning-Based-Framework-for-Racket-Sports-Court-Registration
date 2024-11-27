import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.metrics import IoU
from segmentation_models_pytorch.utils.losses import DiceLoss
import torch.nn
from torch.utils.data import DataLoader
import utils
import model_utils as mu
import os.path as osp
import time
import logging
import cv2
import pandas as pd
import dataset_utils as du
from collections import defaultdict
from segmentation_models_pytorch.utils.functional import iou
logger = logging.getLogger(__name__)


class TennisModel:
    def __init__(self,
                 dataset,
                 experiment_id,
                 experiment_name,
                 model_name,
                 input_height,
                 input_width,
                 encoder,
                 encoder_weights,
                 freeze_encoder,
                 activation,
                 optimizer,
                 learning_rate,
                 loss_function,
                 batch_size,
                 n_epochs,
                 experiment_path,
                 overlay_opacity,
                 fineTune
                 ):
        self.dataset = dataset
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.input_height = input_height
        self.input_width = input_width
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.freeze_encoder = freeze_encoder
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.experiment_path = experiment_path
        self.fineTune = fineTune
        self.overlay_opacity=overlay_opacity

        self.istrained = False
        self.isloaded = False

        self.logfilehandler = logging.FileHandler(osp.join(experiment_path, "logs", "model_output.log"), 'a')
        self.logfilehandler.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
        logger.addHandler(self.logfilehandler)
        logger.info("-" * 40)
        logger.info(self.__class__.__name__ + " instance created")

    def prepare(self):
        self.prepare_model()
        self.freeze_model()
        self.get_batch_size()
        self.setup_loaders()
        self.setup_train()
        self.setup_progress()

    def prepare_model(self):
        self.model = torch.nn.Sequential()
        self.preprocessing_fn = lambda **kwargs: kwargs

    def get_batch_size(self):
        batch_size_gpu = mu.get_batch_size(
            model=self.model,
            input_shape=(3, self.input_height, self.input_width),
            output_shape=(len(self.dataset), self.input_height, self.input_width),
            dataset_train_size=self.dataset.train_size(),
            dataset_valid_size=self.dataset.valid_size()
        )
        logger.info(f"Gpu batch size: {batch_size_gpu}")
        self.batch_size = int(min(self.batch_size, batch_size_gpu))
        logger.info(f"Batch size after gpu check: {self.batch_size}")

    def freeze_model(self):
        if self.freeze_encoder:
            n_train_param=0
            for child in self.model.encoder.children():
                for param in child.parameters():
                    param.requires_grad = False
                    n_train_param+=1
            logger.info(f"Trainable parameters: {n_train_param}")
            logger.info(f"Non-trainable parameters: {self.get_model_nparameters()-n_train_param}")

    def setup_loaders(self):
        self.dataset.build_train(self.preprocessing_fn)
        self.train_loader = DataLoader(self.dataset.train_dataset, batch_size=self.batch_size, shuffle=True,
                                       drop_last=self.dataset.train_size() % self.batch_size == 1)
        self.valid_loader = DataLoader(self.dataset.valid_dataset, batch_size=self.batch_size, shuffle=False,
                                       drop_last=self.dataset.valid_size() % self.batch_size == 1)
        #self.test_loader = DataLoader(self.dataset.test_dataset, batch_size=1)
        logger.info("Dataset loaders setted up")

    def setup_train(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = None
        self.metrics = []
        self.train_optimizer = None


    def train(self):
        self.training_time = 0
        self.gpu_usage = utils.get_gpu_memory()
        logger.info(f"GPU usage: {utils.get_gpu_memory()}MB")
        self.istrained = True

    def load_best_model(self,checkpoint_filename=None):
        checkpoint=f"best_model.pth" if checkpoint_filename is None else checkpoint_filename
        if not osp.exists(osp.join(self.experiment_path, "checkpoints", checkpoint)):
            logger.error(f"Can't find {checkpoint_filename} checkpoint weights")
            return
        #if not self.isloaded:
        if not self.isloaded or checkpoint_filename is not None:
            self.inference_model = torch.load(osp.join(self.experiment_path, "checkpoints", checkpoint),
                                                  map_location=self.device)
            self.isloaded = True
        #logger.info("Model is loaded into memory")

    def load_model(self,checkpoint_filename):
        self.inference_model = torch.load(osp.join(self.experiment_path, "checkpoints", checkpoint_filename),
                                          map_location=self.device)

    def setup_progress(self):
        self.progression_output = osp.join(self.experiment_path, "Progression")
        utils.create_folder(self.progression_output)

        image_filename = osp.join(self.dataset.x_train_dir, "020.png")
        mask_filename = osp.join(self.dataset.y_train_dir, "020.png")

        self.image_progress = cv2.cvtColor(cv2.imread(image_filename), cv2.COLOR_BGR2RGB)
        image_preprocess = self.preprocessing_fn(image=self.dataset.test_augmentation(image=self.image_progress)['image'])['image']
        self.image_preprocess_tensor = torch.from_numpy(image_preprocess).to(self.device).unsqueeze(0)
        gt_mask_progress = cv2.cvtColor(cv2.imread(mask_filename), cv2.COLOR_BGR2RGB)
        self.image_vis_progress = cv2.resize(self.image_progress, (self.input_width, self.input_height))
        self.gt_mask_vis_progress = du.one_hot_encode(gt_mask_progress,
                                        len(self.dataset))
        self.gt_mask_tensor_progress = torch.from_numpy(utils.to_tensor(self.gt_mask_vis_progress)).to(self.device).unsqueeze(0)

        progression_columns = dict({'epoch': 'int'}, **{c: 'float' for c in self.dataset.class_names})
        self.progression_df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in progression_columns.items()})

    def generate_progress(self,epoch):
        pred_mask_tensor = self.model(self.image_preprocess_tensor)

        pred_mask = self.post_processing(pred_mask_tensor)

        pred_mask_overlay = utils.generate_overlay(pred_mask, self.image_vis_progress, self.overlay_opacity,
                                                   self.dataset.colour_palette)

        gt_mask_vis_overlay = utils.generate_overlay(utils.reverse_one_hot(self.gt_mask_vis_progress),
                                                     self.image_vis_progress, self.overlay_opacity,
                                                     self.dataset.colour_palette)

        fp_mask_overlay = utils.generate_mask_fp(pred_mask, utils.reverse_one_hot(self.gt_mask_vis_progress), self.image_vis_progress,
                                                 self.overlay_opacity,
                                                 self.dataset.colour_palette[:])

        final_image = utils.concat_images(
            [[self.image_vis_progress, gt_mask_vis_overlay], [pred_mask_overlay, fp_mask_overlay]])
        save_path = osp.join(self.progression_output, f"{epoch}.png")
        cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

        progression_partition = defaultdict(float)

        for i, c in enumerate(self.dataset.class_names):
            progression_partition[c] = iou(pred_mask_tensor, self.gt_mask_tensor_progress,
                                           ignore_channels=[n for n in range(self.dataset.n_classes) if
                                                            n != i]).detach().squeeze().cpu().numpy()
        self.progression_df = self.progression_df.append(
            dict({'epoch': epoch}, **progression_partition),
            ignore_index=True)

        self.progression_df.to_csv(osp.join(self.experiment_path, "iou_progression.csv"), index=False)

    def unload(self):
        self.isloaded = False
        #del self.inference_model

    def post_processing(self, tensor):
        pass

    def inference(self, image,checkpoint_filename=None):
        pass

    def test(self):
        pass

    def get_results(self, config):
        self.dataset.get_results(config)
        config["batch_size"] = self.batch_size
        config["n_parameters"] = self.get_model_nparameters()
        config["model_size"] = self.get_model_size()
        config["gpu_usage"] = self.gpu_usage
        config["training_time"] = self.training_time

    def get_model_size(self):
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        return round((param_size + buffer_size) / 1024 ** 2, 3)

    def get_model_nparameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def __del__(self):
        del self.dataset
        logger.removeHandler(self.logfilehandler)


class Deeplabv3plusModel(TennisModel):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

    def prepare_model(self):
        self.stride = min(mu.get_stride(self.input_height, self.input_width), 16)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.fineTune:
            # In the next experimentation: BadtoTennis modify me
            self.model = torch.load(osp.join(self.experiment_path, "checkpoints", f"best_model.pth"),
                                    map_location=self.device)
        else:
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.encoder,
                encoder_weights=self.encoder_weights,
                classes=len(self.dataset),
                activation=self.activation,
                encoder_output_stride=self.stride
            )

        self.preprocessing_fn = mu.get_preprocessing(
            smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights))
        logger.info("Model created")
        logger.info(f"Output stride: {self.stride}")


    def setup_train(self):
        if self.loss_function == "WeightedDiceLoss":
            self.loss = DiceLoss()
        if self.loss_function == "jaccard_loss":
            self.loss = mu.JaccardLoss()

        if self.loss_function == "dice_weighted":
            self.loss = mu.DiceLoss(ignore_channels=[n for n in range(self.dataset.n_classes) if n != 10 and n != 12])
        self.metrics = [
            IoU(threshold=0.5),
        ]

        if self.optimizer == "adam":
            self.train_optimizer = torch.optim.Adam([
                dict(params=self.model.parameters(), lr=self.learning_rate)
            ])

        if self.optimizer == "sgd":
            self.train_optimizer = torch.optim.SGD(
                params=self.model.parameters(), lr=self.learning_rate
            )
        logger.info("Training setted up")

    def train(self):
        if self.istrained and not self.fineTune:
            logger.error("Model already trained")
            return
        try:
            train_epoch = mu.TrainEpoch(
                self.model,
                loss=self.loss,
                metrics=self.metrics,
                optimizer=self.train_optimizer,
                device=self.device,
                verbose=True,
            )

            valid_epoch = mu.ValidEpoch(
                self.model,
                loss=self.loss,
                metrics=self.metrics,
                device=self.device,
                verbose=True,
            )
            logger.info(f"Started training experiment {self.experiment_id}")
            self.train_miou = 0.0
            self.valid_miou = 0.0
            self.train_WeightedDiceLoss = 10000.0
            self.valid_WeightedDiceLoss = 10000.0
            self.train_logs_list, self.valid_logs_list = [], []
            self.train_logs_iter_list, self.valid_logs_iter_list = [], []
            start_time = time.time()
            try:
                for i in range(1, self.n_epochs + 1):
                    # Perform training & validation
                    logger.info("")
                    logger.info(
                        f"Experiment: {self.experiment_id}_{self.experiment_name}. Model: {self.model_name}. Epoch: {i}")
                    train_logs, train_iter_logs = train_epoch.run(self.train_loader)
                    print(train_logs)
                    logger.info(
                        f"Train epoch {i} results: WeightedDiceLoss={train_logs['WeightedDiceLoss']:.5f}, miou={train_logs['iou_score']:.5f}")
                    valid_logs, valid_iter_logs = valid_epoch.run(self.valid_loader)
                    logger.info(
                        f"Valid epoch {i} results: WeightedDiceLoss={valid_logs['WeightedDiceLoss']:.5f}, miou={valid_logs['iou_score']:.5f}")
                    self.train_logs_list.append(train_logs)
                    self.valid_logs_list.append(valid_logs)
                    self.train_logs_iter_list.extend(train_iter_logs)
                    self.valid_logs_iter_list.extend(valid_iter_logs)
                    # Save model if a better val IoU score is obtained

                    if self.valid_WeightedDiceLoss >= valid_logs['WeightedDiceLoss']:
                        self.valid_miou = valid_logs['iou_score']
                        self.train_miou = train_logs['iou_score']
                        self.valid_WeightedDiceLoss = valid_logs['WeightedDiceLoss']
                        self.train_WeightedDiceLoss = train_logs['WeightedDiceLoss']
                        torch.save(self.model, osp.join(self.experiment_path, "checkpoints", f"best_model.pth"))
                        self.saved_epoch = i
                        logger.info("")
                        logger.info('Better results obtained and saved')

                    # if i == self.n_epochs and i == self.saved_epoch:
                    #     torch.save(self.model, osp.join(self.experiment_path, "checkpoints", f"last_epoch_model.pth"))
                    #     logger.info("")
                    #     logger.info(f'Saved last epoch model weights')
                    
                    self.generate_progress(i)
                    if i>self.n_epochs-12:
                        torch.save(self.model, osp.join(self.experiment_path, "checkpoints", f"best_model_{i}.pth"))
            except KeyboardInterrupt:
                pass
            logger.info("")
            super().train()
            self.training_time = int(time.time() - start_time)
            logger.info(f"Traning ended in {self.training_time / 3600} hours")

        except RuntimeError as e:
            logger.error("CUDA out of memory error")
            raise Exception("GPU Memory insufficient for this experiment")

    def test(self):
        self.load_best_model()
        test_epoch = smp.utils.train.ValidEpoch(
            self.inference_model,
            loss=self.loss,
            metrics=self.metrics,
            device=self.device,
            verbose=True,
        )
        logger.info("")
        logger.info("Evaluation on Test Data: ")
        test_logs = test_epoch.run(self.test_loader)
        self.test_miou = test_logs['iou_score']
        self.test_WeightedDiceLoss = test_logs['WeightedDiceLoss']
        logger.info(
            f"Test results: WeightedDiceLoss={test_logs['WeightedDiceLoss']:.5f}, miou={test_logs['iou_score']:.5f}")

    def post_processing(self, tensor):
        return utils.transpose_reverse_one_hot(tensor.detach().squeeze().cpu().numpy())

    def inference(self, image, preprocessing=True,checkpoint_filename=None):
        self.load_best_model(checkpoint_filename)
        if preprocessing:
            image = self.preprocessing_fn(image=self.dataset.test_augmentation(image=image)['image'])['image']
        x_tensor = torch.from_numpy(image).to(self.device).unsqueeze(0)
        pred_mask = self.inference_model(x_tensor)
        return pred_mask

    def get_results(self, config):
        super().get_results(config)
        config["stride"] = self.stride
        config["saved_epoch"] = self.saved_epoch
        config["train_miou"] = self.train_miou
        config["valid_miou"] = self.valid_miou
        #config["test_miou"] = self.test_miou
        config["train_WeightedDiceLoss"] = self.train_WeightedDiceLoss
        config["valid_WeightedDiceLoss"] = self.valid_WeightedDiceLoss
        #config["test_WeightedDiceLoss"] = self.test_WeightedDiceLoss


class Deeplabv3plusdynamicModel(Deeplabv3plusModel):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

    def setup_train(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss1 = DiceLoss()
        self.loss2 = DiceLoss(
            ignore_channels=[n for n in range(self.dataset.n_classes) if n not in self.dataset.back_ids])
        # self.loss3 = mu.JaccardLoss()

        self.metrics = [
            IoU(threshold=0.5),
        ]
        if self.optimizer == "adam":
            self.train_optimizer = torch.optim.Adam([
                dict(params=self.model.parameters(), lr=self.learning_rate)
            ])

        if self.optimizer == "sgd":
            self.train_optimizer = torch.optim.SGD(
                params=self.model.parameters(), lr=self.learning_rate
            )
        logger.info("Training setted up")

    def train(self):
        if self.istrained and not self.fineTune:
            logger.error("Model already trained")
            return
        try:
            train_epoch1 = mu.TrainEpoch(
                self.model,
                loss=self.loss1,
                metrics=self.metrics,
                optimizer=self.train_optimizer,
                device=self.device,
                verbose=True,
            )

            valid_epoch1 = mu.ValidEpoch(
                self.model,
                loss=self.loss1,
                metrics=self.metrics,
                device=self.device,
                verbose=True,
            )
            train_epoch2 = mu.TrainEpoch(
                self.model,
                loss=self.loss2,
                metrics=self.metrics,
                optimizer=self.train_optimizer,
                device=self.device,
                verbose=True,
            )

            valid_epoch2 = mu.ValidEpoch(
                self.model,
                loss=self.loss2,
                metrics=self.metrics,
                device=self.device,
                verbose=True,
            )
            # train_epoch3 = mu.TrainEpoch(
            #     self.model,
            #     loss=self.loss3,
            #     metrics=self.metrics,
            #     optimizer=self.train_optimizer,
            #     device=self.device,
            #     verbose=True,
            # )
            #
            # valid_epoch3 = mu.ValidEpoch(
            #     self.model,
            #     loss=self.loss3,
            #     metrics=self.metrics,
            #     device=self.device,
            #     verbose=True,
            # )
            logger.info(f"Started training experiment {self.experiment_id}")
            self.train_miou = 0.0
            self.valid_miou = 0.0
            self.train_WeightedDiceLoss = 10000.0
            self.valid_WeightedDiceLoss = 10000.0
            self.train_logs_list, self.valid_logs_list = [], []
            self.train_logs_iter_list, self.valid_logs_iter_list = [], []
            loss_start,loss_length=list(map(int,self.loss_function.split("-")))
            start_time = time.time()
            try:
                for i in range(1, self.n_epochs + 1):
                    # Perform training & validation
                    logger.info("")
                    logger.info(
                        f"Experiment: {self.experiment_id}_{self.experiment_name}. Model: {self.model_name}. Epoch: {i}")
                    if i in range(loss_start, loss_start + loss_length):
                        train_logs, train_iter_logs = train_epoch2.run(self.train_loader)
                        logger.info(
                            f"Train epoch {i} results: WeightedDiceLoss2={train_logs['WeightedDiceLoss']:.5f}, miou={train_logs['iou_score']:.5f}")
                        valid_logs, valid_iter_logs = valid_epoch2.run(self.valid_loader)
                        logger.info(
                            f"Valid epoch {i} results: WeightedDiceLoss2={valid_logs['WeightedDiceLoss']:.5f}, miou={valid_logs['iou_score']:.5f}")
                    # elif i in range(32, 41):
                    #     train_logs, train_iter_logs = train_epoch3.run(self.train_loader)
                    #     logger.info(
                    #         f"Train epoch {i} results: WeightedDiceLoss3={train_logs['WeightedDiceLoss']:.5f}, miou={train_logs['iou_score']:.5f}")
                    #     valid_logs, valid_iter_logs = valid_epoch3.run(self.valid_loader)
                    #     logger.info(
                    #         f"Valid epoch {i} results: WeightedDiceLoss3={valid_logs['WeightedDiceLoss']:.5f}, miou={valid_logs['iou_score']:.5f}")
                    else:
                        train_logs, train_iter_logs = train_epoch1.run(self.train_loader)
                        logger.info(
                            f"Train epoch {i} results: WeightedDiceLoss1={train_logs['WeightedDiceLoss']:.5f}, miou={train_logs['iou_score']:.5f}")
                        valid_logs, valid_iter_logs = valid_epoch1.run(self.valid_loader)
                        logger.info(
                            f"Valid epoch {i} results: WeightedDiceLoss1={valid_logs['WeightedDiceLoss']:.5f}, miou={valid_logs['iou_score']:.5f}")
                    self.train_logs_list.append(train_logs)
                    self.valid_logs_list.append(valid_logs)
                    self.train_logs_iter_list.extend(train_iter_logs)
                    self.valid_logs_iter_list.extend(valid_iter_logs)
                    # Save model if a better val IoU score is obtained

                    if self.valid_WeightedDiceLoss >= valid_logs['WeightedDiceLoss']:
                        self.valid_miou = valid_logs['iou_score']
                        self.train_miou = train_logs['iou_score']
                        self.valid_WeightedDiceLoss = valid_logs['WeightedDiceLoss']
                        self.train_WeightedDiceLoss = train_logs['WeightedDiceLoss']
                        torch.save(self.model, osp.join(self.experiment_path, "checkpoints", f"best_model.pth"))
                        self.saved_epoch = i
                        logger.info("")
                        logger.info('Better results obtained and saved')

                    # if i == self.n_epochs and i == self.saved_epoch:
                    #     torch.save(self.model, osp.join(self.experiment_path, "checkpoints", f"last_epoch_model.pth"))
                    #     logger.info("")
                    #     logger.info(f'Saved last epoch model weights')

                    self.generate_progress(i)
                    if i > self.n_epochs - 15:
                        torch.save(self.model, osp.join(self.experiment_path, "checkpoints", f"best_model_{i}.pth"))
            except KeyboardInterrupt:
                pass
            logger.info("")
            super(Deeplabv3plusModel, self).train()
            self.training_time = int(time.time() - start_time)
            logger.info(f"Traning ended in {self.training_time / 3600} hours")

        except RuntimeError as e:
            logger.error("CUDA out of memory error")
            raise Exception("GPU Memory insufficient for this experiment")

    def test(self):
        self.load_best_model()
        test_epoch = smp.utils.train.ValidEpoch(
            self.inference_model,
            loss=self.loss1,
            metrics=self.metrics,
            device=self.device,
            verbose=True,
        )
        logger.info("")
        logger.info("Evaluation on Test Data: ")
        test_logs = test_epoch.run(self.test_loader)
        self.test_miou = test_logs['iou_score']
        self.test_WeightedDiceLoss = test_logs['WeightedDiceLoss']
        logger.info(
            f"Test results: WeightedDiceLoss={test_logs['WeightedDiceLoss']:.5f}, miou={test_logs['iou_score']:.5f}")


class UnetModel(Deeplabv3plusdynamicModel):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

    def prepare_model(self):
        # self.stride = utils.get_stride(self.input_height, self.input_width)
        self.model = smp.Unet(
            encoder_name=self.encoder,
            encoder_weights=self.encoder_weights,
            classes=len(self.dataset),
            activation=self.activation,
        )
        self.preprocessing_fn = mu.get_preprocessing(
            smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights))
        logger.info("Model created")
        # logger.info(f"Output stride: {self.stride}")

    def get_results(self, config):
        super(Deeplabv3plusModel, self).get_results(config)
        # config["stride"] = self.stride
        config["saved_epoch"] = self.saved_epoch
        config["train_miou"] = self.train_miou
        config["valid_miou"] = self.valid_miou
        config["test_miou"] = self.test_miou
        config["train_WeightedDiceLoss"] = self.train_WeightedDiceLoss
        config["valid_WeightedDiceLoss"] = self.valid_WeightedDiceLoss
        config["test_WeightedDiceLoss"] = self.test_WeightedDiceLoss


class UnetplusplusModel(UnetModel):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

    def prepare_model(self):
        # self.stride = utils.get_stride(self.input_height, self.input_width)
        self.model = smp.UnetPlusPlus(
            encoder_name=self.encoder,
            encoder_weights=self.encoder_weights,
            classes=len(self.dataset),
            activation=self.activation,
        )
        self.preprocessing_fn = mu.get_preprocessing(
            smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights))
        logger.info("Model created")
        # logger.info(f"Output stride: {self.stride}")


class LinknetModel(UnetModel):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

    def prepare_model(self):
        # self.stride = utils.get_stride(self.input_height, self.input_width)
        self.model = smp.Linknet(
            encoder_name=self.encoder,
            encoder_weights=self.encoder_weights,
            classes=len(self.dataset),
            activation=self.activation,
        )
        self.preprocessing_fn = mu.get_preprocessing(
            smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights))
        logger.info("Model created")
        # logger.info(f"Output stride: {self.stride}")


class PanModel(Deeplabv3plusModel):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

    def prepare_model(self):
        self.stride = min(mu.get_stride(self.input_height, self.input_width), 16)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.fineTune:
            self.model = torch.load(osp.join(self.experiment_path, "checkpoints", f"best_model.pth"),
                                    map_location=self.device)
        else:
            self.model = smp.PAN(
                encoder_name=self.encoder,
                encoder_weights=self.encoder_weights,
                classes=len(self.dataset),
                activation=self.activation,
                encoder_output_stride=self.stride
            )
        self.preprocessing_fn = mu.get_preprocessing(
            smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights))
        logger.info("Model created")
        logger.info(f"Output stride: {self.stride}")


class Deeplabv3Model(Deeplabv3plusdynamicModel):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

    def prepare_model(self):
        self.stride = min(mu.get_stride(self.input_height, self.input_width), 16)
        self.model = smp.DeepLabV3(
            encoder_name=self.encoder,
            encoder_weights=self.encoder_weights,
            classes=len(self.dataset),
            activation=self.activation,
        )
        self.preprocessing_fn = mu.get_preprocessing(
            smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights))
        logger.info("Model created")
        logger.info(f"Output stride: {self.stride}")


class ManetModel(UnetModel):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

    def prepare_model(self):
        # self.stride = mu.get_stride(self.input_height, self.input_width)
        self.model = smp.MAnet(
            encoder_name=self.encoder,
            encoder_weights=self.encoder_weights,
            classes=len(self.dataset),
            activation=self.activation,
        )
        self.preprocessing_fn = mu.get_preprocessing(
            smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights))
        logger.info("Model created")
        # logger.info(f"Output stride: {self.stride}")
