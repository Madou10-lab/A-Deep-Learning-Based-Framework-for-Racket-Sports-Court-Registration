import os.path as osp

columns_experiment = {
    'experiment_id': 'Int64',
    'experiment_name': 'str',
    'model_name': 'str',
    'dataset_name': 'str',
    'exploitation_name': 'str',
    'input_height': 'Int64',
    'input_width': 'Int64',
    'shuffle': 'bool',
    'split_ratio': 'float64',
    'augmentation_colour_format': 'str',
    'augmentation_spatial': 'bool',
    'augmentation_colour': 'bool',
    'encoder': 'str',
    'encoder_weights': 'str',
    'freeze_encoder':'bool',
    'activation': 'str',
    'optimizer': 'str',
    'learning_rate': 'float64',
    'loss_function': 'str',
    'batch_size': 'Int64',
    'n_epochs': 'Int64',
    'keep': 'bool',
    'n_classes': 'Int64',
    'stride': 'Int64',
    'training_time': 'float64',
    'gpu_usage': 'float64',
    'model_size': 'float64',
    'n_parameters': 'Int64',
    'train_size': 'Int64',
    'valid_size': 'Int64',
    'test_size': 'Int64',
    'train_miou': 'float64',
    'train_dice_loss': 'float64',
    'valid_miou': 'float64',
    'valid_dice_loss': 'float64',
    'test_miou': 'float64',
    'test_dice_loss': 'float64',
    'saved_epoch': 'Int64',
    'inference_time': 'float64',
    'inference_gpu_usage': 'float64',
    'video_fps': 'float64',
    'video_elapsed_time': 'float64',
    'experiment_elapsed_time': 'float64',
    'completed': 'bool',
    'error': 'str'
}

dataset_keys = [
    'experiment_id',
    'dataset_name',
    #'split_type',
    'input_height',
    'input_width',
    'shuffle',
    'augmentation_colour_format',
    'augmentation_spatial',
    'augmentation_colour',
    'split_ratio',
    'labels_path',
    'dataset_path',
    #'test_vid_ids',
    'experiment_path',
]

model_keys = [
    'experiment_id',
    'experiment_name',
    'model_name',
    'input_height',
    'input_width',
    'encoder',
    'encoder_weights',
    'freeze_encoder',
    'activation',
    'optimizer',
    'learning_rate',
    'loss_function',
    'batch_size',
    'n_epochs',
    'experiment_path',
    'fineTune',
    'overlay_opacity'
]

exploitation_keys = [
    'experiment_id',
    'dataset_name',
    'exploitation_name',
    'input_height',
    'input_width',
    'overlay_opacity',
    'experiment_path',
]

def get_experiment_path(config):
    dirname = f"{config['experiment_id']}_{config['model_name']}_{config['dataset_name']}_{config['experiment_name']}_{config['n_epochs']}"
    return osp.join(config["experiments_path"], dirname)

dataset_variants=["Main","onlyFull","onlyShuttle","onlyBack","onlyNet","courtZones","onlyLines","fullZones"]
#split_variants=["all","green","red","exist","doubles","occlusion"]
model_variants= ["DeepLabV3Plus","DeepLabV3Plusdynamic","DeepLabV3","Unet","UnetPlusPlus","PAN","LinkNet","MANet"]

encoders= ["resnet101","resnet50","resnet18","resnet34","inceptionv4","inceptionresnetv2","xception","vgg16",
    "vgg19","mit_b1","mit_b3","resnet152","efficientnet-b2","efficientnet-b5","timm-tf_efficientnet_lite4"]
weights_type= ["imagenet"]
activations= ["sigmoid","softmax2d"]
optimizers= ["adam","sgd"]
loss_functions= ["dice_loss","jaccard_loss","dice_loss_dyn","dice_weighted"]
colour_formats= ["none","gray","hsv",'bgr']
