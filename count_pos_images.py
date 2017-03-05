#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 19:42:32 2017

@author: ping133
"""

import pandas as pd
import sys

if __name__ == '__main__':
    examsMetadataFilename = sys.argv[1]
    imagesCrosswalkFilename = sys.argv[2]

    # Pandas converts columns that have missing values to double
    metadata = pd.read_csv(examsMetadataFilename, sep="\t", na_values='.')
    # Read the image metadata
    images = pd.read_csv(imagesCrosswalkFilename, sep="\t", na_values='.')
    
    # Count the number of positive images
    numImagesPosL = 0
    metadataL = metadata.loc[metadata.cancerL == 1]
    for i in range(0, metadataL.shape[0]):
        # Read the images corresponding to the cancer status
        imageMetadata = images.loc[(images.subjectId == metadataL.subjectId.iloc[i]) & \
                                   (images.examIndex == metadataL.examIndex.iloc[i]) & \
                                   (images.laterality == "L")]
        numImagesPosL = numImagesPosL + imageMetadata.shape[0]
    numImagesPosR = 0
    metadataR = metadata.loc[metadata.cancerR == 1]
    for i in range(0, metadataR.shape[0]):
        # Read the images corresponding to the cancer status
        imageMetadata = images.loc[(images.subjectId == metadataR.subjectId.iloc[i]) & \
                                   (images.examIndex == metadataR.examIndex.iloc[i]) & \
                                   (images.laterality == "R")]
        numImagesPosR = numImagesPosR + imageMetadata.shape[0]
    
    numImagesPos = numImagesPosL + numImagesPosR    
        
    # Count them
    print "num. total images = {}" .format(images.shape[0])
    print "num. positive images = {}, {} %" .format(numImagesPos,
                                  float(numImagesPos)/images.shape[0]*100)
