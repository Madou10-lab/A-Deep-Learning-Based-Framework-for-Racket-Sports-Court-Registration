import albumentations as album

import torch
import typing as t
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.utils import functional as F2
from segmentation_models_pytorch.base.modules import Activation

from time import sleep
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.metrics import IoU
import torch.nn
import sys
from tqdm import tqdm as tqdm
from segmentation_models_pytorch.utils.meter import AverageValueMeter


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)


def get_stride(height, width):
    stride = 2
    while height % stride == 0 and width % stride == 0 and stride < 64:
        stride *= 2
    return stride // 2


def get_batch_size(
        model: nn.Module,
        input_shape: t.Tuple[int, int, int],
        output_shape: t.Tuple[int, int, int],
        dataset_train_size: int,
        dataset_valid_size: int,
        max_batch_size=16,
        num_iterations: int = 5,
) -> int:
    max_batch_size = min(max_batch_size, dataset_train_size // 2, dataset_valid_size)
    device = torch.device("cuda")
    model.to(device)
    model.train(True)
    optimizer = torch.optim.Adam(model.parameters())
    inputs, targets, loss = None, None, None

    # print("Test batch size")
    batch_size = 2
    while True:
        if batch_size > max_batch_size:
            batch_size = batch_size // 2
            break
        try:
            for _ in range(num_iterations):
                # dummy inputs and targets
                inputs = torch.rand(*(batch_size, *input_shape), device=device)
                targets = torch.rand(*(batch_size, *output_shape), device=device)
                outputs = model(inputs)
                loss = F.mse_loss(targets, outputs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # print(f"\tTesting batch size {batch_size}")
            # print(get_gpu_memory())
            batch_size *= 2
            sleep(3)
        except RuntimeError:
            # print(f"\tOOM at batch size {batch_size}")
            # print(get_gpu_memory())
            batch_size //= 2
            # print(traceback.format_exc())
            break
    del model, optimizer, inputs, targets, loss, device
    torch.cuda.empty_cache()
    # print(f"Final batch size {batch_size}")
    # print(get_gpu_memory())
    return batch_size


class DiceLoss(base.Loss):
    def __init__(self, eps=1.0, beta=1.0, omega=0.8, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.omega=omega

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        l1=1 - F2.f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=None,
        )

        l2=1 - F2.f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


        l_res = self.omega * l1 + (1-self.omega) * l2

        return l_res

class JaccardLoss(base.Loss):
    def __init__(self, eps=1.0, activation=None, ignore_channels=None, **kwargs):
        self._name = "dice_loss"
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self._name= "dice_loss"

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F2.jaccard(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )

    @property
    def __name__(self):
        return self._name

class BoundaryLoss(base.Loss):
    def __init__(self, eps=1.0, beta=1.0,  activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        l1=1 - F2.f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=None,
        )

        l2=1 - F2.f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )

        return l2


class DiceLossMixAverage(base.Loss):
    def __init__(self, eps=1.0, beta=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        l1 = 1 - F2.f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=None,
        )

        l2 = 1 - F2.f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )

        l_res = ( l1 + l2 ) / 2

        return l_res

class DiceLossMixAdd(base.Loss):
    def __init__(self, eps=1.0, beta=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        l1 = 1 - F2.f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=None,
        )

        l2 = 1 - F2.f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )

        l_res = l1 + l2

        return l_res

class TrainEpoch(smp.utils.train.TrainEpoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            device=device,
            verbose=verbose,
            optimizer=optimizer
        )

    def run(self, dataloader):

        self.on_epoch_start()

        logs_iterations = []
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(
                dataloader,
                desc=self.stage_name,
                file=sys.stdout,
                disable=not (self.verbose),
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                #loss_logs = {self.loss.__name__: loss_meter.mean}
                loss_logs = {self.loss.name: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs_iterations.append(metrics_logs | loss_logs)
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)
        return logs, logs_iterations

class ValidEpoch(smp.utils.train.ValidEpoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            device=device,
            verbose=verbose,
        )

    def run(self, dataloader):

        self.on_epoch_start()

        logs_iterations = []
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(
                dataloader,
                desc=self.stage_name,
                file=sys.stdout,
                disable=not (self.verbose),
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                #loss_logs = {self.loss.__name__: loss_meter.mean}
                #loss_logs = {"dice_loss": loss_meter.mean}
                loss_logs = {self.loss.name: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs_iterations.append(metrics_logs | loss_logs)
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)
        return logs, logs_iterations
