# referenced from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import time
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# special forward function for fivecrop-ed data
# use in validation
def fivecrop_forward(inputs, model, weights_ratio=[0.15, 0.15, 0.15, 0.15, 0.4]):
    tl, tr, bl, br, ct = inputs
    outputs_tl = model(tl) * weights_ratio[0]
    outputs_tr = model(tr) * weights_ratio[1]
    outputs_bl = model(bl) * weights_ratio[2]
    outputs_br = model(br) * weights_ratio[3]
    outputs_ct = model(ct) * weights_ratio[4]
    # as the center crop usually captures the key element of the face, it deserves greater weight
    outputs = outputs_tl + outputs_tr + outputs_bl + outputs_br + outputs_ct
    return outputs


# train the model wrt to their unique pretrained output shape
#   notice that the dataloaders is a dictionary like {'train': train_loader, 'val': val_loader}
def train_model(model, dataloaders, criterion,
                optimizer, scheduler=None, val_func=fivecrop_forward,
                num_epochs=25, is_inception=False, half=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            top1_corrects = 0
            top5_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if half:
                    if phase == 'train':
                        inputs = inputs.half().to(device)
                    else:
                        if type(inputs) is list:
                            inputs = [img_crop_batch.half().to(device) for img_crop_batch in inputs]
                        else:
                            inputs = inputs.half().to(device)
                else:
                    if phase == 'train':
                        inputs = inputs.to(device)
                    else:
                        if type(inputs) is list:
                            inputs = [img_crop_batch.to(device) for img_crop_batch in inputs]
                        else:
                            inputs = inputs.to(device)
                labels = labels.to(device)  # no need to half the labels

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    # as FiveCrop are used for val_ds, special eval steps are required
                    elif phase == 'val' and val_func:
                        outputs = val_func(inputs, model)
                        loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.topk(outputs, 5, dim=1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                if phase == 'train':
                    running_loss += loss.item() * inputs.size(0)
                else:
                    if type(inputs) is list:
                        running_loss += loss.item() * inputs[0].size(0)
                    else:
                        running_loss += loss.item() * inputs.size(0)
                top1_corrects += torch.sum(preds[:, 0] == labels.data)
                top5_corrects += torch.sum(preds == labels.unsqueeze(1))

            if scheduler is not None and phase == "train":
                scheduler.step()

            dataset_size = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / dataset_size
            epoch_top1acc = top1_corrects.double() / dataset_size
            epoch_top5acc = top5_corrects.double() / dataset_size

            print(f'{phase} Loss: {epoch_loss:.4f}  '
                  f'Top-1 Acc: {epoch_top1acc:.4f} ({top1_corrects.double()}/{dataset_size}) '
                  f'Top-5 Acc: {epoch_top5acc:.4f} ({top5_corrects.double()}/{dataset_size}) ')

            # deep copy the model
            if phase == 'val' and epoch_top1acc > best_acc:
                best_acc = epoch_top1acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append((epoch_top1acc, epoch_top5acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# freeze all the parameters if extracting the feature map produced by the model
#   the last output layer can be reinitialized for training to fit the local data (see initialize_model())
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# reinitialize the last layer of model for fine-tuning wrt to their unique output shape
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# help configuring the optimizer to only train the reinitialized layer
def config_optim(optim_class, model_ft, feature_extract=True, **kwargs):
    params_to_update = model_ft.parameters()
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                # print("\t", name)

    # Observe that all parameters are being optimized
    kwargs['params'] = params_to_update
    optimizer_ft = optim_class(**kwargs)

    return optimizer_ft


IMG_LABELS = sorted([str(i) for i in range(1000)])
LABELS_TRANSLATOR = dict([(str(idx), int(label)) for idx, label in enumerate(IMG_LABELS)])


def evaluation(model, test_img, augment=None):
    model.eval()
    if isinstance(test_img, str):  # if is file_path, read the image
        with Image.open(test_img) as f:
            test_img = f  # load the PIL image
    if augment:
        test_img = augment(test_img)
    with torch.no_grad():
        outputs = fivecrop_forward(test_img, model)
        _, preds = torch.topk(outputs, 1, dim=1)
        preds = [LABELS_TRANSLATOR[str(pred)] for pred in preds.flatten().tolist()]
    return preds


if __name__ == "__main__":
    print(f'using {device}')
