#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 23:32:33 2017

@author: paolo
"""

import pandas as pd
import sys
import random
import os

if __name__ == '__main__':
    examsMetadataFilename = sys.argv[1]
    imagesCrosswalkFilename = sys.argv[2]
    imageLabelsTempFilename = sys.argv[3]
    outputDir = sys.argv[4]
    random.seed(float(sys.argv[5]))

    metadata = pd.read_csv(examsMetadataFilename, sep="\t", na_values='.')
    images = pd.read_csv(imagesCrosswalkFilename, sep="\t", na_values='.')
    labels = pd.read_csv(imageLabelsTempFilename, sep=" ", na_values='.',
                         header=None)
    labels.columns = ['filename', 'cancer']
    imagesPos = set(labels.filename[labels.cancer == 1])
    imagesNeg = set(labels.filename[labels.cancer == 0])
    print "num. positive images = {}" .format(len(imagesPos))
    print "num. negative images = {}" .format(len(imagesNeg))
    imagesNeg = set(random.sample(imagesNeg,
                                  int(len(imagesPos))))
    print "after sampling: num. negative images = {}" .format(len(imagesNeg))
    imagesNew = imagesPos.union(imagesNeg)
    # Update the image crosswalk and the exams metadata
    images = images.loc[images.filename.isin(imagesNew)]
    metadata = metadata.loc[metadata.subjectId.isin(set(images.subjectId))]
    # Write the new exams metadata train and images crosswalk train
    metadata.to_csv(os.path.join(outputDir, "exams_metadata_train_UNDER.tsv"),
                    sep="\t", na_rep='.', index=False, header=True)
    images.to_csv(os.path.join(outputDir, "images_crosswalk_train_UNDER.tsv"),
                  sep="\t", na_rep='.', index=False, header=True)
