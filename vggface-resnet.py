from model.vgg_face import VGGFaceResNet
from data.augment import augmentation_train, augmentation_test
from data.data import get_ds, get_loader
from helper.setup import config_optim
from helper.process import train_model
from helper.eval import fivecrop_forward
import torch.cuda as cuda
import torch.optim as optim
import torch.nn as nn
import torch

if __name__ == "__main__":
    # data preparation
    BATCH_SIZE = 32
    IMG_RESIZE = 224
    TRAIN_CROP_SIZE, VAL_CROP_SIZE = 48, 48
    train_ds = get_ds('train', transformation=augmentation_train(TRAIN_CROP_SIZE, IMG_RESIZE,
                                                                 preprocess=False, normalized=False))
    train_loader = get_loader(train_ds, BATCH_SIZE, shuffle=True)
    val_ds = get_ds('val', transformation=augmentation_test(VAL_CROP_SIZE, IMG_RESIZE,
                                                            preprocess=False, normalized=False))
    val_loader = get_loader(val_ds, BATCH_SIZE, shuffle=True)
    dataloaders = {'train':  train_loader, 'val': val_loader}

    # prepare model
    NUM_OF_CLASSES = 1000
    model = VGGFaceResNet(3, NUM_OF_CLASSES)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(r"trials/vggface-resnet-7505.pth"))
    model = model.half().to(device)

    # training settings
    criterion = nn.CrossEntropyLoss()
    optimizer = config_optim(optim.SGD, model_ft=model,  feature_extract=False,
                             lr=1e-5, momentum=0.9, weight_decay=0.01)
    scheduler_lambda_func = lambda epoch: 1 if epoch < 5 else (0.5 if epoch < 10 else (0.25 if epoch < 15 else 0.1))
    # scheduler_lambda_func = lambda epoch: 1 if epoch < 10 else 0.1
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_lambda_func)
    val_func = lambda inputs, net: fivecrop_forward(inputs, net)

    # train the model
    epochs = 20
    cuda.empty_cache()
    model_ft, hist = train_model(model, dataloaders, criterion,
                                 optimizer, scheduler=scheduler, val_func=val_func,
                                 num_epochs=epochs, is_inception=False, half=True)
    torch.save(model_ft.state_dict(), f'vggface-resnet.pth')
