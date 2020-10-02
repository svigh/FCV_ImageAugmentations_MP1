import numpy as np
import cv2
import tkinter.filedialog
import os
import sys
import time
import argparse

_DEFINE_USE_TKINTER = False

IMPLEMENTED_OPERATIONS = ["rotate", "abs_tint", "tint", "flip", "blur", "noise"]
DEFAULT_ROTATION_DEGREES = 15
DEFAULT_BLUR_KSIZE = 3

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_dir", dest="input_dir_path", action="store", required=True,
		help="The input directory that contains all the images to be augmented")
	parser.add_argument("--config_file", dest="config_file_path", action="store", required=True,
		help="The input config file that contains the augmentations to be added")

	args = parser.parse_args()
	return args

def get_augmentations_from_file(config_file):
	augments = []
	# Create a list of dictionaries:
	# [ {operation: operationName0: params: [param0, param1]},
	#	{operation: operationName1: params: [param0]},.. ]
	try:
		with open(config_file, "r") as config:
			for line in config:
				line_contents_list = line.split()
				# If line is empty skip it
				if not line_contents_list:
					continue

				operation = line_contents_list[0].lower()
				params = [ param.lower() for param in line_contents_list[1:] ]

				if operation not in IMPLEMENTED_OPERATIONS:
					print("Operation %s not implemented, ignoring" % (operation))
					continue

				operation_dict = {}
				operation_dict["operation"] = operation
				operation_dict["params"] = params
				augments.append(operation_dict)
	except FileNotFoundError as e:
		print("Error at reading config file %s\n" % (config_file), str(e))
	return augments

def apply_tint(image_path, params, use_absolute_values=False):
	new_image = cv2.imread(image_path)
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

	print("Applied TINT on channels %s on %s" % (params, image_path))
	return new_image

def apply_rotate(image_path, params):
	new_image = cv2.imread(image_path)
	height, width = new_image.shape[1::-1]
	try:
		degrees = int(params[0])
	except IndexError:
		degrees = DEFAULT_ROTATION_DEGREES
		params.append(str(degrees))

	image_center = tuple(np.array([height, width]) / 2)
	rotation_matrix = cv2.getRotationMatrix2D(image_center, degrees, 1.0)

	# Find the bounding box that contains the rotated image
	# so that the corners dont get cut
	abs_cos = abs(rotation_matrix[0][0])
	abs_sin = abs(rotation_matrix[0][1])
	bound_w = int(height * abs_sin + width * abs_cos)
	bound_h = int(height * abs_cos + width * abs_sin)
	rotation_matrix[0][2] += bound_w/2 - image_center[0]
	rotation_matrix[1][2] += bound_h/2 - image_center[1]

	new_image = cv2.warpAffine(new_image, rotation_matrix, (bound_w, bound_h), flags=cv2.INTER_LINEAR)

	print("Applied ROTATE %s degrees on %s" % (degrees, image_path))
	return new_image

def apply_blur(image_path, params):
	try:
		k_size = int(params[0])
		if k_size % 2 == 0:
			k_size -= 1
			params[0] = str(k_size)	# Update the params to generate a correct output name
	except IndexError:
		print("Got blur without intensity parameter, defaulting to 3")
		k_size = DEFAULT_BLUR_KSIZE
		params.append(str(k_size))	# Update the params to generate a correct output name
	new_image = cv2.imread(image_path)

	cv2.GaussianBlur(new_image, (k_size, k_size), 0, dst=new_image)
	return new_image

def apply_noise(image_path, params):
	new_image = cv2.imread(image_path)

	height, width, depth = new_image.shape
	noise_array = np.random.uniform(low=0.7, high=1, size=(height, width, depth))

	aux_image = new_image * noise_array
	new_image = aux_image.astype("uint8")

	return new_image


def apply_flip(image_path, params):
	new_image = cv2.imread(image_path)
	axis = []

	for flip_axis in params:
		if flip_axis.lower() == "vertical":
			new_image = cv2.flip(new_image, 0)
			axis.append(flip_axis)
		if flip_axis.lower() == "horizontal":
			new_image = cv2.flip(new_image, 1)
			axis.append(flip_axis)

	print("Applied FLIP %s on %s" % (axis, image_path))
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

def apply_augment(input_image_path, augment):
	if augment["operation"] == "rotate":
		new_image = apply_rotate(input_image_path, augment["params"])
	elif augment["operation"] == "abs_tint":
		new_image = apply_tint(input_image_path, augment["params"], use_absolute_values=True)
	elif augment["operation"] == "tint":
		new_image = apply_tint(input_image_path, augment["params"], use_absolute_values=False)
	elif augment["operation"] == "flip":
		new_image = apply_flip(input_image_path, augment["params"])
	elif augment["operation"] == "blur":
		new_image = apply_blur(input_image_path, augment["params"])
	elif augment["operation"] == "noise":
		new_image = apply_noise(input_image_path, augment["params"])
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
		for augment in augments:
			new_image = apply_augment(image_path, augment)

			new_image_name = image_basename_no_extension + "_" + augment["operation"] + \
			  ("-" + "-".join(augment["params"]) if len(augment["params"]) > 0 else "") + "_" + str(augment_index) + "." + image_extension
			new_image_path = str(os.path.join(output_directory,new_image_name))

			print("\tWriting augmentation to %s" % (new_image_path))
			cv2.imwrite(new_image_path, new_image)
			augment_index += 1

def main():
	config_file_path = None
	input_dir_path = None

	if _DEFINE_USE_TKINTER:
		config_file_path = tkinter.filedialog.askopenfilename(title="CONFIG FILE")
		input_dir_path = tkinter.filedialog.askdirectory(title="INPUT DIRECTORY")
	else:
		args = parse_args()
		config_file_path = args.config_file_path
		input_dir_path = args.input_dir_path

	augments = get_augmentations_from_file(config_file_path)
	apply_augments(input_dir_path, augments)


if __name__ == '__main__':
	main()

