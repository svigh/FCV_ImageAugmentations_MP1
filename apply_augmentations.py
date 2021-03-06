import numpy as np
import cv2
import tkinter.filedialog
import os
import sys
import time
import argparse
import math
from enum import Enum

DEFAULT_ROTATION_DEGREES = 15
DEFAULT_BLUR_KSIZE = 3
DEFAULT_LOWER_NOISE_LIMIT = 0.7
DEFAULT_UPPER_NOISE_LIMIT = 1
DEFAULT_BRIGHTNESS_INTENSITY = 1.5
DEFAULT_RESCALE_WIDTH = 256
DEFAULT_RESCALE_HEIGHT= 256

class ROTATION_MODES(Enum):
	KEEP_ORIGINAL_SIZE = 1
	KEEP_CORNERS = 2
	CROP_INWARD = 3


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_dir", dest="input_dir_path", action="store",
		help="The input directory that contains all the images to be augmented")
	parser.add_argument("--config_file", dest="config_file_path", action="store",
		help="The input config file that contains the augmentations to be added")

	args = parser.parse_args()
	return args

def get_augmentations_from_file(config_file):
	augments = []
	# Create a list of lists of dictionaries:
	# [ [{operation: operationName0: params: [param0, param1]},
	#	{operation: operationName1: params: [param0]},.. ], [..., ...] ]
	try:
		with open(config_file, "r") as config:
			for line in config:
				augments_chain = []
				for inline_operation in line.split(";"):
					line_contents_list = inline_operation.split()
					# If line is empty skip it
					if not line_contents_list:
						continue

					operation = line_contents_list[0].lower()
					params = [ param.lower() for param in line_contents_list[1:] ]

					operation_dict = {}
					operation_dict["operation"] = operation
					operation_dict["params"] = params
					augments_chain.append(operation_dict)
				if len(augments_chain) > 0:
					augments.append(augments_chain)
	except FileNotFoundError as e:
		print("Error at reading config file %s\n" % (config_file), str(e))
	return augments

def apply_tint(image, params, use_absolute_values=False):
	new_image = image
	height, width = new_image.shape[1::-1]
	channels = []

	for channel in params:
		if "blue" in channel.lower():
			try:
				value = int(channel.replace("blue", ""))
			except ValueError:
				print("Got tint param %s but expected blueXX" % (channel))
				continue

			if use_absolute_values:
				if value > 255:
					new_image[:,:,0] = 255
				else:
					new_image[:,:,0] = value
			else:
				temp_array = np.array(new_image[:,:,0], dtype=np.uint16)
				temp_array += value
				np.clip(temp_array, 0, 255, out=new_image[:,:,0])
		elif "green" in channel.lower():
			try:
				value = int(channel.replace("green", ""))
			except ValueError:
				print("Got tint param %s but expected greenXX" % (channel))
				continue

			if use_absolute_values:
				if value > 255:
					new_image[:,:,1] = 255
				else:
					new_image[:,:,1] = value
			else:
				temp_array = np.array(new_image[:,:,1], dtype=np.uint16)
				temp_array += value
				np.clip(temp_array, 0, 255, out=new_image[:,:,1])
		elif "red" in channel.lower():
			try:
				value = int(channel.replace("red", ""))
			except ValueError:
				print("Got tint param %s but expected redXX" % (channel))
				continue

			if use_absolute_values:
				if value > 255:
					new_image[:,:,2] = 255
				else:
					new_image[:,:,2] = value
			else:
				temp_array = np.array(new_image[:,:,2], dtype=np.uint16)
				temp_array += value
				np.clip(temp_array, 0, 255, out=new_image[:,:,2])
		else:
			continue

	return new_image

# Rotate the image with crop or no crop
def apply_rotate(image, params, crop_type=ROTATION_MODES.KEEP_ORIGINAL_SIZE):
	new_image = image
	height, width = new_image.shape[1::-1]

	# Get the params set
	try:
		degrees = int(params[0])
	except IndexError:
		degrees = DEFAULT_ROTATION_DEGREES
		params.append(str(degrees))

	image_center = tuple(np.array([height, width]) / 2)
	rotation_matrix = cv2.getRotationMatrix2D(image_center, degrees, 1.0)

	# Good calculator that explains stuff
	# https://www.calculator.net/right-triangle-calculator.html

	if crop_type == ROTATION_MODES.KEEP_ORIGINAL_SIZE:
		bound_w, bound_h = height, width
	elif crop_type == ROTATION_MODES.KEEP_CORNERS:
		# Find the bounding box that contains the rotated image
		# so that the corners dont get cut
		abs_cos = abs(rotation_matrix[0][0])
		abs_sin = abs(rotation_matrix[0][1])
		bound_h = int(height * abs_sin + width * abs_cos)
		bound_w = int(height * abs_cos + width * abs_sin)
	elif crop_type == ROTATION_MODES.CROP_INWARD:	# TODO: find direct formula for bounding box
		# Get the bounding box so that the height and width of the
		# original image are the hypothenuse of each corner triangle
		# and get the short edge length to eliminate from the bounding
		# box around the image
		abs_cos = abs(rotation_matrix[0][0])
		abs_sin = abs(rotation_matrix[0][1])
		bound_h = int(height * abs_sin + width * abs_cos)
		bound_w = int(height * abs_cos + width * abs_sin)
		bound_w -= int(width * abs_sin) * 2
		bound_h -= int(height * abs_sin) * 2

	rotation_matrix[0][2] += bound_w/2 - image_center[0]
	rotation_matrix[1][2] += bound_h/2 - image_center[1]

	new_image = cv2.warpAffine(new_image, rotation_matrix, (bound_w, bound_h), flags=cv2.INTER_LINEAR)

	return new_image

def apply_rescale(image, params):
	new_image = image
	height, width = 0, 0

	# Get the params set
	try:
		width = int(params[0])
	except IndexError:
		width = DEFAULT_RESCALE_WIDTH
		params.append(str(width))
	try:
		height = int(params[1])
	except IndexError:
		height = DEFAULT_RESCALE_HEIGHT
		params.append(str(height))

	new_image = cv2.resize(new_image, (width, height))

	return new_image

# Blur the image with a gausian kernel of size k (bigger kernel for fuzzier result)
def apply_blur(image, params):
	new_image = image
	k_size = 0

	# Get the params set
	try:
		k_size = int(params[0])
		if k_size % 2 == 0:
			k_size -= 1
			params[0] = str(k_size)	# Update the params to generate a correct output name
	except IndexError:
		print("Got blur without intensity parameter, defaulting to 3")
		k_size = DEFAULT_BLUR_KSIZE
		params.append(str(k_size))	# Update the params to generate a correct output name

	cv2.GaussianBlur(new_image, (k_size, k_size), 0, dst=new_image)
	return new_image

# Multiply each channel of each pixel with a random amount
def apply_noise(image, params):
	new_image = image
	height, width, depth = new_image.shape

	# Get the params set
	try:
		lower_noise_limit = float(params[0])
	except IndexError:
		lower_noise_limit = DEFAULT_LOWER_NOISE_LIMIT
		params.append(str(lower_noise_limit))
	try:
		upper_noise_limit = float(params[1])
	except IndexError:
		upper_noise_limit = DEFAULT_UPPER_NOISE_LIMIT
		params.append(str(upper_noise_limit))

	noise_array = np.random.uniform(
		low=lower_noise_limit,
		high=upper_noise_limit,
		size=(height, width, depth)
	)

	# Avoid overflow
	aux_image = new_image * noise_array
	np.clip(aux_image, 0, 255, out=new_image)

	return new_image

# TODO: maximum amount of brightness for each pixel
# should be when any channel reaches 255
# Multiply each channel with intensity
def apply_brighten(image, params):
	new_image = image
	intensity = 0.0

	# Get the params set
	try:
		intensity = float(params[0])
	except IndexError:
		intensity = DEFAULT_BRIGHTNESS_INTENSITY
		params.append(str(intensity))

	aux_image = new_image * intensity
	np.clip(aux_image, 0, 255, out=new_image)

	return new_image

# Flips the image on vertical or horizontal axis
def apply_flip(image, params):
	new_image = image
	axis = []

	# Get the params set
	if len(params) == 0:
		params.append("horizontal")

	for flip_axis in params:
		if flip_axis.lower() == "vertical":
			new_image = cv2.flip(new_image, 0)
			axis.append(flip_axis)
		if flip_axis.lower() == "horizontal":
			new_image = cv2.flip(new_image, 1)
			axis.append(flip_axis)

	return new_image

def create_output_dir(output_directory):
	try:
		os.mkdir(output_directory)
	except FileExistsError:
		output_dir_increment = 1
		while os.path.exists(output_directory + "_" + str(output_dir_increment)):
			output_dir_increment += 1
		try:
			print("Output directory exists, moving it to Output_" + str(output_dir_increment))
			os.rename(output_directory, output_directory + "_" + str(output_dir_increment))
		except OSError as e:
			print(str(e))
			exit(1)
		try:
			os.mkdir(output_directory)
		except OSError as e:
			print(str(e))
			exit(1)

def apply_augment(image, augment):
	# Organizing the functions corresponding to each augment string
	augments_dictionary = {
		"rotate" : apply_rotate,
		"rotate_crop" : lambda image, params : apply_rotate(image, params, crop_type=ROTATION_MODES.CROP_INWARD),
		"rotate_keep_size" : lambda image, params : apply_rotate(image, params, crop_type=ROTATION_MODES.KEEP_ORIGINAL_SIZE),
		"rotate_resize" : lambda image, params : apply_rotate(image, params, crop_type=ROTATION_MODES.KEEP_CORNERS),
		"tint" : apply_tint,
		"abs_tint" : lambda image, params : apply_tint(image, params, use_absolute_values=True),
		"rescale" : apply_rescale,
		"flip" : apply_flip,
		"blur" : apply_blur,
		"noise" : apply_noise,
		"brighten" : apply_brighten
	}

	try:
		new_image = augments_dictionary[augment["operation"]](image, augment["params"])
	except KeyError:
		print("Operation %s not implemented, skipping..." % (augment["operation"]))
		raise

	return new_image

def apply_augments(input_directory, augments, output_directory="Output"):
	print("Applying augments list", augments)

	# Create output dir and move old output dirs
	if len(augments) > 0:
		create_output_dir(output_directory)

	# Start applying augments
	augment_index = 0
	for image in os.listdir(input_directory):
		image_path = os.path.join(input_directory, image)
		image_basename_no_extension, image_extension = image.split(".")

		print("\nAugmenting image %s" % (image_path))
		for augment_chain in augments:
			if len(augment_chain):
				new_image = cv2.imread(image_path)
				try:
					for augment in augment_chain:
						new_image = apply_augment(new_image, augment)
				except KeyError:	# In case the augment is not in the augments dictionary
					continue

				# For each augmentations chain (which is a list of dictionaries)
				# we get each dicts operation and params (if they exist) and create the string
				new_image_name = image_basename_no_extension + \
								"_" + ("---".join([augment["operation"] + \
									("-" + "-".join(augment["params"]) if len(augment["params"]) > 0 else "")
								for augment in augment_chain])) + \
								"_" + str(augment_index) + "." + image_extension

				new_image_path = str(os.path.join(output_directory,new_image_name))

				print("\tWriting %s augmentation to %s" % (augment["operation"], new_image_path))
				cv2.imwrite(new_image_path, new_image)
				augment_index += 1

def main():
	args = parse_args()

	config_file_path = None
	input_dir_path = None

	if args.config_file_path is not None:
		config_file_path = args.config_file_path
	else:
		config_file_path = tkinter.filedialog.askopenfilename(title="CONFIG FILE")

	if args.input_dir_path is not None:
		input_dir_path = args.input_dir_path
	else:
		input_dir_path = tkinter.filedialog.askdirectory(title="INPUT DIRECTORY")

	augments = get_augmentations_from_file(config_file_path)
	for line in augments:
		print(line)

	# sys.exit(0)
	apply_augments(input_dir_path, augments)

if __name__ == '__main__':
	main()

