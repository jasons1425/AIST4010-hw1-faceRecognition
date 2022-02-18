from helper.eval import fivecrop_forward
import torch
import time
import copy


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def half_input(inputs, device, half):
    if type(inputs) is list:
        if half:
            inputs = [img_crop_batch.half().to(device) for img_crop_batch in inputs]
        else:
            inputs = [img_crop_batch.to(device) for img_crop_batch in inputs]
    else:
        if half:
            inputs = inputs.half().to(device)
        else:
            inputs = inputs.to(device)
    return inputs


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
                inputs = half_input(inputs, device, half)
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
