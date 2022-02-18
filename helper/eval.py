from PIL import Image
import torch


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


IMG_LABELS = sorted([str(i) for i in range(1000)])
LABELS_TRANSLATOR = dict([(str(idx), int(label)) for idx, label in enumerate(IMG_LABELS)])


def evaluation(model, test_img, augment=None,
               weights_ratio=[0.15, 0.15, 0.15, 0.15, 0.4]):
    model.eval()
    if isinstance(test_img, str):  # if is file_path, read the image
        with Image.open(test_img) as f:
            test_img = f  # load the PIL image
    if augment:
        test_img = augment(test_img)
    with torch.no_grad():
        outputs = fivecrop_forward(test_img, model, weights_ratio=weights_ratio)
        _, preds = torch.topk(outputs, 1, dim=1)
        preds = [LABELS_TRANSLATOR[str(pred)] for pred in preds.flatten().tolist()]
    return preds
