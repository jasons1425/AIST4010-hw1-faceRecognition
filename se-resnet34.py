from model.se_resnet import se_resnet34
from data.augment import augmentation_train, augmentation_test
from data.data import get_ds, get_loader
from model.ft_helper import train_model, config_optim, fivecrop_forward
import torch.cuda as cuda
import torch.optim as optim
import torch.nn as nn
import torch

if __name__ == "__main__":
    # data preparation
    BATCH_SIZE = 64
    IMG_RESIZE = 224
    TRAIN_CROP_SIZE, VAL_CROP_SIZE = 48, 48
    train_ds = get_ds('train', transformation=augmentation_train(TRAIN_CROP_SIZE, IMG_RESIZE,
                                                                 preprocess=False))
    train_loader = get_loader(train_ds, BATCH_SIZE, shuffle=True)
    val_ds = get_ds('val', transformation=augmentation_test(VAL_CROP_SIZE, IMG_RESIZE,
                                                            preprocess=False))
    val_loader = get_loader(val_ds, BATCH_SIZE, shuffle=True)
    dataloaders = {'train':  train_loader, 'val': val_loader}

    # prepare model
    NUM_OF_CLASSES = 1000
    model = se_resnet34(NUM_OF_CLASSES)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model = model.half().to(device)

    # training settings
    criterion = nn.CrossEntropyLoss()
    optimizer = config_optim(optim.SGD, model_ft=model,  feature_extract=False,
                             lr=0.01, momentum=0.9, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.9)
    val_func = lambda inputs, net: fivecrop_forward(inputs, net)


    # train the model
    epochs = 10
    cuda.empty_cache()
    model_ft, hist = train_model(model, dataloaders, criterion,
                                 optimizer, scheduler, val_func=val_func,
                                 num_epochs=epochs, is_inception=False, half=True)
    torch.save(model_ft.state_dict(), f'se-resnet34.pth')