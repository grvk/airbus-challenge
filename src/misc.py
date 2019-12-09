from torch import argmax, nn

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


class CrossEntropyLoss2D(nn.CrossEntropyLoss):
    """
    See official documentation of nn.CrossEntropyLoss

    The only difference is that this loss accepts optional normalization
    coefficient to deal with vanishing/exploding gradients
    """

    def __init__(self, normalization_coef=1, **kwargs):
        self.norm_coef = normalization_coef
        super(CrossEntropyLoss2D, self).__init__(**kwargs)

    def forward(self, prediction, target):
        """
        Args:
            prediction(:obj:`torch.Tensor`) - predicted values of the form:
                (number of images, number of classes, height, width)
            target(:obj:`torch.Tensor`) - expected dense classification of
                one of the two forms:
                (number of images, 1, height, width) or
                (number of images, height, width)
        return:
            Total loss for all of the images in the batch
        """
        if len(target.size()) == 4:
            target = target.squeeze(1)

        res = super(CrossEntropyLoss2D, self).forward(prediction, target)
        return res if self.norm_coef == 1 else res * self.norm_coef
