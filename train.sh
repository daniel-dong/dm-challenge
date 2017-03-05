#!/bin/bash

EXAMS_METADATA_FILENAME="/metadata/exams_metadata.tsv"
IMAGES_CROSSWALK_FILENAME="/metadata/images_crosswalk.tsv"

PREPROCESS_DIRECTORY="/preprocessedData"
PREPROCESS_IMAGES_DIRECTORY="$PREPROCESS_DIRECTORY/images"
PREPROCESS_METADATA_DIRECTORY="$PREPROCESS_DIRECTORY/metadata"

TRAIN_IMAGES_DIRECTORY="/scratch/trainPatches"
VAL_IMAGES_DIRECTORY="/scratch/valPatches"
TRAIN_METADATA_DIRECTORY="/scratch/metadata"
LMDB_DIRECTORY="/scratch/lmdb"

MODELSTATE_DIRECTORY="/modelState"

TRAIN_SIZE=0.9
RANDOM_SEED=(123456 234567 345678 456789 567890) # Used to split training/test

# Count the positive images
python count_pos_images.py $EXAMS_METADATA_FILENAME \
	$IMAGES_CROSSWALK_FILENAME

#=================================================================================
for ITER_NUMBER in {0..4}; do

	echo "Model $ITER_NUMBER"

	echo "Generate training/val sets"

	rm -rf $TRAIN_METADATA_DIRECTORY/*
	rm -rf $LMDB_DIRECTORY

	mkdir -p $TRAIN_METADATA_DIRECTORY
	mkdir -p $LMDB_DIRECTORY
	mkdir -p $TRAIN_IMAGES_DIRECTORY
	#mkdir -p $VAL_IMAGES_DIRECTORY

	echo "Splitting the challenge training set into training and validation sets"
	python generate_train_val_sets.py $EXAMS_METADATA_FILENAME \
		$IMAGES_CROSSWALK_FILENAME \
		$TRAIN_METADATA_DIRECTORY \
		$TRAIN_SIZE \
        ${RANDOM_SEED[ITER_NUMBER]}

	echo "Generating temporary labels"
	python generate_image_labels.py $TRAIN_METADATA_DIRECTORY/exams_metadata_train.tsv \
		$TRAIN_METADATA_DIRECTORY/images_crosswalk_train.tsv \
		$TRAIN_METADATA_DIRECTORY/image_labels_tmp.txt

#	echo "Generating temporary labels"
#	python generate_image_labels.py $EXAMS_METADATA_FILENAME \
#		$IMAGES_CROSSWALK_FILENAME \
#		$TRAIN_METADATA_DIRECTORY/image_labels_tmp.txt

	echo "Undersampling negative training images"
	python undersample_neg_images.py $TRAIN_METADATA_DIRECTORY/exams_metadata_train.tsv \
		$TRAIN_METADATA_DIRECTORY/images_crosswalk_train.tsv \
		$TRAIN_METADATA_DIRECTORY/image_labels_tmp.txt \
		$TRAIN_METADATA_DIRECTORY \
        ${RANDOM_SEED[ITER_NUMBER]}

	echo "Extract train patches"
	python extract_patches.py $TRAIN_METADATA_DIRECTORY/exams_metadata_train_UNDER.tsv \
		$TRAIN_METADATA_DIRECTORY/images_crosswalk_train_UNDER.tsv \
		$PREPROCESS_IMAGES_DIRECTORY \
		$TRAIN_IMAGES_DIRECTORY \
		$TRAIN_METADATA_DIRECTORY/images_crosswalk_train_PATCHES.tsv

	echo "Extract val patches"
	python extract_patches.py $TRAIN_METADATA_DIRECTORY/exams_metadata_val.tsv \
		$TRAIN_METADATA_DIRECTORY/images_crosswalk_val.tsv \
		$PREPROCESS_IMAGES_DIRECTORY \
		$VAL_IMAGES_DIRECTORY \
		$TRAIN_METADATA_DIRECTORY/images_crosswalk_val_PATCHES.tsv

	echo "Generating image labels for TRAIN PATCHES"
	python generate_image_labels.py $TRAIN_METADATA_DIRECTORY/exams_metadata_train_UNDER.tsv \
		$TRAIN_METADATA_DIRECTORY/images_crosswalk_train_PATCHES.tsv \
		$TRAIN_METADATA_DIRECTORY/image_labels_train_PATCHES.txt
	sed -i 's/.dcm/.png/g' $TRAIN_METADATA_DIRECTORY/image_labels_train_PATCHES.txt

	echo "Generating image labels for TRAIN IMAGES"
	python generate_image_labels.py $TRAIN_METADATA_DIRECTORY/exams_metadata_train_UNDER.tsv \
		$TRAIN_METADATA_DIRECTORY/images_crosswalk_train_UNDER.tsv \
		$TRAIN_METADATA_DIRECTORY/image_labels_train.txt
	sed -i 's/.dcm/.png/g' $TRAIN_METADATA_DIRECTORY/image_labels_train.txt

	echo "Generating image labels for VAL PATCHES"
	python generate_image_labels.py $TRAIN_METADATA_DIRECTORY/exams_metadata_val.tsv \
	        $TRAIN_METADATA_DIRECTORY/images_crosswalk_val_PATCHES.tsv \
	        $TRAIN_METADATA_DIRECTORY/image_labels_val_PATCHES.txt
	sed -i 's/.dcm/.png/g' $TRAIN_METADATA_DIRECTORY/image_labels_val_PATCHES.txt

	echo "Generating image labels for VAL IMAGES"
	python generate_image_labels.py $TRAIN_METADATA_DIRECTORY/exams_metadata_val.tsv \
	        $TRAIN_METADATA_DIRECTORY/images_crosswalk_val.tsv \
	        $TRAIN_METADATA_DIRECTORY/image_labels_val.txt
	sed -i 's/.dcm/.png/g' $TRAIN_METADATA_DIRECTORY/image_labels_val.txt

	echo "Generating LMDB train"
	convert_imageset --backend=lmdb \
	    --shuffle \
	    --gray=false \
	    $TRAIN_IMAGES_DIRECTORY/ \
	    $TRAIN_METADATA_DIRECTORY/image_labels_train_PATCHES.txt \
	    $LMDB_DIRECTORY/train

	echo "Generating mean image for backgroud substraction"
	compute_image_mean $LMDB_DIRECTORY/train $MODELSTATE_DIRECTORY/mean_train_$ITER_NUMBER.binaryproto

	echo "Generating LMDB val"
	convert_imageset --backend=lmdb \
	    --shuffle \
	    --gray=false \
	    $VAL_IMAGES_DIRECTORY/ \
	    $TRAIN_METADATA_DIRECTORY/image_labels_val_PATCHES.txt \
	    $LMDB_DIRECTORY/val

	python convnet.py $TRAIN_METADATA_DIRECTORY/images_crosswalk_train_PATCHES.tsv \
        $TRAIN_METADATA_DIRECTORY/images_crosswalk_val_PATCHES.tsv \
        $TRAIN_METADATA_DIRECTORY/images_crosswalk_val.tsv \
        $TRAIN_METADATA_DIRECTORY/exams_metadata_val.tsv \
        $TRAIN_METADATA_DIRECTORY/image_labels_train.txt \
		$TRAIN_IMAGES_DIRECTORY \
        $TRAIN_METADATA_DIRECTORY/image_labels_val.txt \
        $VAL_IMAGES_DIRECTORY \
		$MODELSTATE_DIRECTORY \
		$ITER_NUMBER

done

echo "done"
