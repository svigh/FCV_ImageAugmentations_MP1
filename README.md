# FCV_ImageAugmentations_MP1

Run command: python3 apply_augmentations.py --input_dir=Input --config_file=config

Expected output:
Operation #rotate not implemented, ignoring
Operation #rotate not implemented, ignoring
Operation # not implemented, ignoring
Operation #asd not implemented, ignoring
Operation #flip not implemented, ignoring
Operation #flip not implemented, ignoring
Operation #flip not implemented, ignoring
Applying augments list [{'operation': 'tint', 'params': ['blue0']}, {'operation': 'tint', 'params': ['greenasd0']}, {'operation': 'abs_tint', 'params': ['red0']}]
Output directory exists, moving it to Output_3

Augmenting image Input/image1.jpg
Applied TINT on channels ['blue0'] on Input/image1.jpg
	Writing augmentation to Output/image1_tint-blue0_0.jpg
Got tint param greenasd0 but expected greenXX
Applied TINT on channels ['greenasd0'] on Input/image1.jpg
	Writing augmentation to Output/image1_tint-greenasd0_1.jpg
Applied TINT on channels ['red0'] on Input/image1.jpg
	Writing augmentation to Output/image1_abs_tint-red0_2.jpg

Augmenting image Input/image3.jpg
Applied TINT on channels ['blue0'] on Input/image3.jpg
	Writing augmentation to Output/image3_tint-blue0_3.jpg
Got tint param greenasd0 but expected greenXX
Applied TINT on channels ['greenasd0'] on Input/image3.jpg
	Writing augmentation to Output/image3_tint-greenasd0_4.jpg
Applied TINT on channels ['red0'] on Input/image3.jpg
	Writing augmentation to Output/image3_abs_tint-red0_5.jpg

Augmenting image Input/image2.jpg
Applied TINT on channels ['blue0'] on Input/image2.jpg
	Writing augmentation to Output/image2_tint-blue0_6.jpg
Got tint param greenasd0 but expected greenXX
Applied TINT on channels ['greenasd0'] on Input/image2.jpg
	Writing augmentation to Output/image2_tint-greenasd0_7.jpg
Applied TINT on channels ['red0'] on Input/image2.jpg
	Writing augmentation to Output/image2_abs_tint-red0_8.jpg

