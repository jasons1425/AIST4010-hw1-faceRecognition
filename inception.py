import torch.cuda
from data.augment import augmentation_train, augmentation_test
from data.data import get_ds, get_loader
from model.ft_helper import train_model, initialize_model, config_optim
import torch.optim as optim
import torch.nn as nn

if __name__ == "__main__":
    # data preparation
    BATCH_SIZE = 64
    IMG_RESIZE = 299 # inception expects (299, 299) sized img
    CROP_SIZE = 32
    train_ds = get_ds('train', transformation=augmentation_train(CROP_SIZE, IMG_RESIZE))
    train_loader = get_loader(train_ds, BATCH_SIZE, shuffle=True)
    val_ds = get_ds('val', transformation=augmentation_test(CROP_SIZE, IMG_RESIZE))  # inception expects (299, 299) sized img
    val_loader = get_loader(val_ds, BATCH_SIZE, shuffle=True)
    dataloaders = {'train':  train_loader, 'val': val_loader}

    # get pretrained model
    NUM_OF_CLASSES = 1000
    model_name = "inception"
    model_ft, input_size = initialize_model(model_name, NUM_OF_CLASSES, feature_extract=False, use_pretrained=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_ft = model_ft.to(device)

    # get training settings
    criterion = nn.CrossEntropyLoss()
    optimizer = config_optim(optim.SGD, model_ft=model_ft,  feature_extract=False,
                             lr=0.001, momentum=0.9, weight_decay=0.01)

    # train the model
    epochs = 40
    model_ft, hist = train_model(model_ft, dataloaders, criterion,
                                 optimizer, num_epochs=epochs,
                                 is_inception=(model_name == "inception"))
    torch.save(model_ft.state_dict(), f'inception_ft.pth')





