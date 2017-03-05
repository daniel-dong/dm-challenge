#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:43:07 2017

@author: paolo
"""

import os
import sys
import cv2
import pandas as pd


def gen_cropped_image(im, rx, cx, size, images, nameString, imageDir,
                      imgMetaNew, outputImageDir):

    imgMetaNew['orig'] = imgMetaNew.filename
    imgMetaNew.filename = imgMetaNew.filename.replace(
        ".png", "_" + nameString + ".png")

    croppedImage = cv2.resize(im[rx[0]:rx[1], cx[0]:cx[1], :], (size, size))
    params = list()
    params.append(cv2.IMWRITE_PNG_COMPRESSION)
    params.append(0)
    cv2.imwrite(outputImageDir + '/' +
                imgMetaNew.filename, croppedImage, params)

    images = images.append(imgMetaNew)

    return images


if __name__ == '__main__':
    examsMetadataFilename = sys.argv[1]
    imagesCrosswalkFilename = sys.argv[2]
    imageDir = sys.argv[3]
    outputImageDir = sys.argv[4]
    imagesCrosswalkPatches = sys.argv[5]

    # Read the metadata of the training set
    metadata = pd.read_csv(examsMetadataFilename, sep="\t", na_values='.')
    images = pd.read_csv(imagesCrosswalkFilename, sep="\t", na_values='.')
    images.loc[:, 'filename'] = images.filename.str.replace('.dcm', '.png')

    # Take the metadata corresponding to the images subjects
    subjectIds = set(images.subjectId)
    metadata = metadata.loc[metadata.subjectId.isin(subjectIds)]

    # Calculate the ratio of positive and negative images in the training set
    numImagesPosL = 0
    metaL = metadata.loc[metadata.cancerL == 1]
    for i in range(0, metaL.shape[0]):
        # Read the images corresponding to the cancer status
        imgMeta = images.loc[(images.subjectId == metaL.subjectId.iloc[i]) & (
            images.examIndex == metaL.examIndex.iloc[i]) &
            (images.laterality == "L")]
        numImagesPosL = numImagesPosL + imgMeta.shape[0]
    numImagesPosR = 0
    metaR = metadata.loc[metadata.cancerR == 1]
    for i in range(0, metaR.shape[0]):
        # Read the images corresponding to the cancer status
        imgMeta = images.loc[(images.subjectId == metaR.subjectId.iloc[i]) & (
            images.examIndex == metaR.examIndex.iloc[i]) &
            (images.laterality == "R")]
        numImagesPosR = numImagesPosR + imgMeta.shape[0]

    numImagesPos = numImagesPosL + numImagesPosR
    totImages = images.shape[0]
    print "n. positive images = {}" .format(numImagesPos)
    print "n. total images = {}" .format(totImages)

    imagesNew = pd.DataFrame()

    for i in range(0, metadata.shape[0]):
        for laterality in ["L", "R"]:
            imgMeta = images.loc[
                    (images.subjectId == metadata.subjectId.iloc[i]) &
                    (images.examIndex == metadata.examIndex.iloc[i]) &
                    (images.laterality == laterality)
                    ]

            if (laterality == "L") and (metadata.cancerL.iloc[i] == 1):
                isCancer = True
            elif (laterality == "R") and (metadata.cancerR.iloc[i] == 1):
                isCancer = True
            else:
                isCancer = False
            # Select the corresponding images
            for j in range(0, imgMeta.shape[0]):

                print "filename: {}" .format(imgMeta.iloc[j].filename)
                # Load current image
                im = cv2.imread(imageDir + "/" + imgMeta.iloc[j].filename)
                rows, cols = im.shape[:2]
#                # Apply CLAHE
#                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#                clahe = cv2.createCLAHE()
#                im = clahe.apply(im)
#                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

                # Generate the cropped images
                rx = int(rows / 3)
                cx = int(cols / 3)
                imgMetaNew = imgMeta.iloc[j].copy()
                imagesNew = gen_cropped_image(
                    im, [0, 2 * rx], [0, 2 * cx], 227, imagesNew, "crop1",
                    imageDir, imgMetaNew, outputImageDir
                    )
                imgMetaNew = imgMeta.iloc[j].copy()
                imagesNew = gen_cropped_image(
                    im, [rx + 2, rows], [0, 2 * cx], 227, imagesNew, "crop2",
                    imageDir, imgMetaNew, outputImageDir
                    )
                imgMetaNew = imgMeta.iloc[j].copy()
                imagesNew = gen_cropped_image(
                    im, [0, 2 * rx], [cx + 2, cols], 227, imagesNew, "crop3",
                    imageDir, imgMetaNew, outputImageDir
                    )
                imgMetaNew = imgMeta.iloc[j].copy()
                imagesNew = gen_cropped_image(
                    im, [rx + 2, rows], [cx + 2, cols], 227, imagesNew,
                    "crop4", imageDir, imgMetaNew, outputImageDir
                    )
                imgMetaNew = imgMeta.iloc[j].copy()
                imagesNew = gen_cropped_image(
                        im, [rows / 2 - rx, rows / 2 + rx],
                        [cols / 2 - cx, cols / 2 + cx], 227, imagesNew,
                        "crop5", imageDir, imgMetaNew, outputImageDir
                        )

    imagesNew.subjectId = imagesNew.subjectId.astype(int)
    imagesNew.examIndex = imagesNew.examIndex.astype(int)

    imagesNew.to_csv(os.path.join(imagesCrosswalkPatches),
                     sep="\t", na_rep='.', index=False, header=True)
