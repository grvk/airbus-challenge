from torch import argmax

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
