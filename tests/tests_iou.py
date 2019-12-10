import unittest
import numpy as np
from src import IntersectionOverUnion


class IoUTests(unittest.TestCase):

	# For all tests below assume that "image belongs to class" = "each pixel
	# of image belongs to class"

	@classmethod
	def assert_ious_equal(cls, received, expected, prediction, mask, val_type):

		# to make sure np.isnan and np.all are always available
		r = np.array(received)
		e = np.array(expected)
		r[np.isnan(r)] = np.ma.masked
		e[np.isnan(e)] = np.ma.masked

		assert np.all(r == e), \
			"Unexpected received {tp}={r}. Expected={e}. Calculated with " \
			"mask:\n{m}\nPrediction:\n{p}\n".format(
				tp=val_type, r=received, e=expected, m=mask, p=prediction)

	def test_one_img_one_class_match(self):
		"""
		One image. Background = 0, not counted => one considered class
		Image belongs to the class.
		Correct prediction and mask
		"""
		mask = np.ones((1, 3, 3))
		prediction = np.ones((1, 2, 3, 3))
		prediction[:, 0, :, :] = 0

		exp_ious = [float('nan'), 1.0]
		exp_iou = 1.0

		iou, ious = IntersectionOverUnion(prediction, mask, [0])
		IoUTests.assert_ious_equal(ious, exp_ious, prediction, mask, "ious")
		IoUTests.assert_ious_equal(iou, exp_iou, prediction, mask, "mean iou")


	def test_two_imgs_two_classes_match(self):
		"""
		Two images. Two classes = [0, 1]
		Image 0 belongs to class 0. Image 1 belongs to class 1.
		Correct prediction and mask
		"""
		mask = np.ones((2, 3, 3))
		mask[0,:, :] = 0

		prediction = np.ones((2, 2, 3, 3))
		prediction[0, 1,:, :] = 0
		prediction[1, 0,:, :] = 0

		exp_ious = [1.0, 1.0]
		exp_iou = 1.0

		iou, ious = IntersectionOverUnion(prediction, mask)
		IoUTests.assert_ious_equal(ious, exp_ious, prediction, mask, "ious")
		IoUTests.assert_ious_equal(iou, exp_iou, prediction, mask, "mean iou")


	def test_two_imgs_two_classes_one_img_wrong(self):
		"""Two images. Three possible classes.
		Both images belong to class 1.
		Prediction for the 2nd image is correct.
		Prediction for the 1st image is incorrect completely (class 0 instead)
		"""
		mask = np.ones((2, 3, 3))

		prediction = np.zeros((2, 3, 3, 3))
		prediction[0, 0, :, :] = 1
		prediction[1, 1, :, :] = 1
		
		exp_ious = [0, .5, float('nan')]
		exp_iou = 0.25

		iou, ious = IntersectionOverUnion(prediction, mask)
		IoUTests.assert_ious_equal(ious, exp_ious, prediction, mask, "ious")
		IoUTests.assert_ious_equal(iou, exp_iou, prediction, mask, "mean iou")

	def test_two_imgs_classes_both_wrong(self):
		"""Two images. Two classes (one excluded) => one class.
		Both images belong to class 0.
		Prediction for the both images is incorrect completely (class 1 instead)
		"""
		mask = np.zeros((2, 3, 3))

		prediction = np.zeros((2, 2, 3, 3))
		prediction[:, 1, :, :] = 1

		exp_ious = [0, float('nan')]
		exp_iou = 0

		iou, ious = IntersectionOverUnion(prediction, mask, [1])
		IoUTests.assert_ious_equal(ious, exp_ious, prediction, mask, "ious")
		IoUTests.assert_ious_equal(iou, exp_iou, prediction, mask, "mean iou")

	def three_images_three_classes_partially_right_predictions(self):
		"""Three images. Four classes, one excluded > three possible classes.
		Image 0: half belongs to class 0, quarter to 1, quarter to 2
		Image 1: half belongs to class 1, quarter to 0, quarter to 2
		Image 2: belongs to class 3 (not counted)
		"""
		mask = np.full((3, 8, 8), None)

		# image 0
		mask[0, :, np.arange(5, 8)] = 0 # 1/2 belongs to class 0; right half
		mask[0, np.arange(0, 4):, np.arange(0, 4)] = 2 # 1/4 to 2; left upper quarter
		mask[0, np.arange(5, 8):, np.arange(4, 8)] = 1 # 1/4 to 1; left bottom quarter

		# image 1
		mask[0, :, np.arange(5, 8)] = 1  # half belongs to class 1; right half
		mask[0, np.arange(0, 4), np.arange(0, 4)] = 2 # 1/4 to 2; left upper quarter
		mask[0, np.arange(5, 8), np.arange(4, 8)] = 0 # 1/4 to 0; left bottom quarter

		# image 2
		mask[2, :, :] = 3 # completely belongs to class 3

		prediction = np.zeros((3, 4, 8, 8))

		# Images 0 and 1
		prediction[[0, 1], 0, np.arange(0, 4), :] = 1 # 1/2 to class 0; upper half
		prediction[[0, 1], 2, np.arange(5, 8), :] = 1 # 1/2 to class 2; bottom half

		# Image 2
		prediction[2, 1, np.arange(0, 4), :] = 1 # 1/2 to class 1; upper half
		prediction[2, 3, np.arange(5, 8), :] = 1 # 1/2 to class 3; bottom half

		# IOU class 0: 
		exp_ious = [1/6.0, 0, 0, np.float('nan')]
		exp_iou = 1/18

		iou, ious = IntersectionOverUnion(prediction, mask)
		IoUTests.assert_ious_equal(iou, exp_iou, prediction, mask, "mean iou")
		IoUTests.assert_ious_equal(ious, exp_ious, prediction, mask, "ious")


