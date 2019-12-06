from torch import Tensor, nn, argmax

class CrossEntropyLossModified(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super(CrossEntropyLossModified, self).__init__(**kwargs)

    def forward(self, input, target):
        return super(CrossEntropyLossModified, self).forward( \
            input, target.squeeze(1))

def BasicIoU(output, target):
    """
    Intersection of Unions (works on binary masks only)

    Args:
        output(:obj:`torch.Tensor`) - predicted segmentation mask
            of shape BATCHES x CHANNELS(always=1) x HEIGHT x WIDTH
        target(:obj:`torch.Tensor`) - expected segmentation mask
            of shape BATCHES x CHANNELS(always=1) x HEIGHT x WIDTH

    Returns:
        IoU = Area of Intersection /
            (output area + input area - area of interscection)
            averaged over batches
    """

    batches_num, _, _, _ = output.size()

    output = output == 1 # convert to bool
    target = target == 1 # convert to bool

    intersctn = torch.sum(output == target, (1,2,3))
    output_area = torch.sum(output, (1,2,3))
    target_area = torch.sum(target, (1,2,3))

    iou_sum = 0
    iou_count = 0
    for i in list(range(batches_num)):
        if (intersctn[0] == 0) and (target_area != 0) or (intersctn[i] != 0):
            ios_sum = intersctn / (output_area + target_area - intersctn)
            iou_count += 1

    return iou_sum / iou_count if iou_count != 0 else 0

def ClassificationAccuracy(output, target):
    """
    ClassificationAccuracy on a given batch

    Args:
        output(:obj:`torch.Tensor`) - predicted segmentation mask
            of shape BATCHES x SCORES FOR DIFFERENT CLASSES
        target(:obj:`torch.Tensor`) - expected segmentation mask
            of shape BATCHES x SCORES FOR DIFFERENT CLASSES

    Returns:
        Classification Accuracy averaged over the batch of images
    """
    predictions = argmax(output.data, 1) # indices of the predicted clases
    correct = (predictions == target).sum().item()
    total = output.size(0)
    return correct / total
