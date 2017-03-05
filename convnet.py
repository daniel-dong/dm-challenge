#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: ping133
"""

import caffe
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
from sklearn.svm import SVC


def find_optimal_C_for_AUC(xTrain, yTrain, xTest, yTest):
    C_2d_range = [10.0 ** i for i in range(-3, 3)]

    accuracy = np.array([])
    auc_score = np.array([])

    for Ctry in C_2d_range:
        clf = SVC(C=Ctry, kernel="linear", probability=True)
        clf.fit(xTrain, yTrain)
        pred = clf.predict(xTest)
        pred_proba = clf.predict_proba(xTest)
        accuracy = np.append(accuracy, np.average(yTest == pred))
        auc_score = np.append(auc_score,
                              roc_auc_score(yTest, pred_proba[:, 1]))
        print "C: {}" .format(Ctry)
        print "accuracy: {}" .format(accuracy[-1])
        print "AUC: {}" .format(auc_score[-1])

    # Extract the optimal parameters to train the final model
    best_auc_idx = np.where(auc_score == max(auc_score))[0]
    best_acc_idx = np.where(accuracy == max(accuracy[best_auc_idx]))[0]
    best_C = C_2d_range[best_acc_idx[0]]

    return best_C


if __name__ == "__main__":
    """
    Extract the 'pool5/7x7_s1' layer features from googleNet model (caffe)
    and train a linear SVM model. Best C parameter is determined from accuracy and AUC
    on the validation test
    """
    imageCrosswalkTrainPatches = sys.argv[1]
    imageCrosswalkValPatches = sys.argv[2]
    imageCrosswalkValImages = sys.argv[3]
    examsMetadataVal = sys.argv[4]

    trainLabelsFilename = sys.argv[5]
    valLabelsFilename = sys.argv[6]

    trainImagesDirectory = sys.argv[7]
    valImagesDirectory = sys.argv[8]
    modelDirectory = sys.argv[9]

    modelNumber = sys.argv[10]

    deployFilename = "/deploy.prototxt"
    modelFilename = "/bvlc_googlenet.caffemodel"
    meanTrainFilename = "/modelState/mean_train_" + modelNumber + \
                        ".binaryproto"

    # Read the training filenames
    imagesTrain = pd.read_csv(imageCrosswalkTrainPatches, sep="\t",
                              na_values=".")
    trainFilenames = np.array(imagesTrain.filename)

    # Read the training labels
    trainLabels = pd.read_csv(trainLabelsFilename, sep=" ", na_values=".",
                              header=None)
    trainLabels.columns = ['filename', 'cancer']

    # Read the validation filenames
    imagesVal = pd.read_csv(imageCrosswalkValPatches, sep="\t",
                         na_values=".")
    valFilenames = np.array(imagesVal.filename)
    
    # Read the validation labels
    valLabels = pd.read_csv(valLabelsFilename, sep=" ", na_values=".",
                            header=None)
                            valLabels.columns = ['filename', 'cancer']

    # Read the test exams metadata
    examsVal = pd.read_csv(examsMetadataVal, sep="\t", na_values=".")

    # Read the model
    net = caffe.Net(deployFilename, modelFilename, caffe.TEST)

    # Read the mean vector
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(meanTrainFilename, 'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))[0]
    meanArray = arr.mean(1).mean(1)

    # Define the preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', meanArray)
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)

    data_blob_shape = net.blobs['data'].data.shape
    data_blob_shape = list(data_blob_shape)

    # Set the input for 5 images at a time
    net.blobs['data'].reshape(5, data_blob_shape[1], data_blob_shape[2],
                              data_blob_shape[3])

    # Generate the training bags
    print "...generating training patches"
    Xtrain = np.zeros((len(trainLabels.filename), 1024 * 5))
    for i in range(len(trainLabels.filename)):
        if (i % 100 == 0):
            print "...extracting features from patches of image {}/{}" \
                  .format(i, len(trainLabels.filename))

        patches = imagesTrain.loc[imagesTrain.orig == trainLabels.filename[i]]

        patchesFilenames = [trainImagesDirectory + "/" + str(x) for x in
                            patches.filename]
        net.blobs['data'].data[...] = map(lambda x: transformer.preprocess(
            'data', caffe.io.load_image(x)), patchesFilenames)
        out = net.forward()

        Xtrain[i, :] = out['pool5/7x7_s1'].reshape(1, -1)

    # Transform the labels in -1, +1
    trainLabels['cancer'].replace(0, -1, inplace=True)

    # Generate validation bags
    print "...generating val. patches"
    Xtest = np.zeros((len(valLabels.filename), 1024 * 5))
    for i in range(0, len(valLabels.filename)):
        print "...extracting features from patches of image {}/{}" \
            .format(i, len(valLabels.filename))
    
        patches = imagesVal.loc[imagesVal.orig == valLabels.filename[i]]
        
        patchesFilenames = [valImagesDirectory + "/" + str(x) for x in
                            patches.filename]
                            net.blobs['data'].data[...] = map(lambda x: transformer.preprocess(
                                'data', caffe.io.load_image(x)), patchesFilenames)
        out = net.forward()
        Xtest[i, :] = np.reshape(out['pool5/7x7_s1'], 1, -1)

    # Transform the labels in -1, +1
    valLabels['cancer'].replace(0, -1, inplace=True)

    # Train the model
    best_C = find_optimal_C_for_AUC(Xtrain, np.array(trainLabels),
                                    Xtest, np.array(valLabels.cancer))

    clf = SVC(C=best_C, kernel="linear", probability=True)
    clf.fit(Xtrain, np.array(trainLabels.cancer))
    pred = clf.predict_proba(Xtrain)
    # Save the model
    print "...saving SVM model {}" .format(modelNumber)
    joblib.dump(clf, modelDirectory + "/SVM_" + modelNumber + ".pkl")

    # Save SVM output and labels (the difference between pos and neg scores can be
    # used to assign weights to multiple models)
    np.savetxt(modelDirectory + "/labels_" + modelNumber + ".csv",
               np.asarray(trainLabels.cancer), delimiter=',')
    np.savetxt(modelDirectory + "/svmpred_" + modelNumber + ".csv",
               np.asarray(pred), delimiter=',')

    # Calculate the scores for the val subjects
    scoresAll = clf.pred_proba(Xtest)
    imagesValNoPatches = pd.read_csv(imageCrosswalkValImages, sep="\t",
                                     na_values=".")

    subjectIds = pd.unique(imagesTest.subjectId)
    scores = dict.fromkeys(subjectIds)
    leftCancer = dict.fromkeys(subjectIds)
    rightCancer = dict.fromkeys(subjectIds)

    for i in range(0, len(subjectIds)):
        print "Id: {}" .format(subjectIds[i])
        
        # Extract the labels
        examsSubject = examsVal.loc[examsVal.subjectId == subjectIds[i], ]
        if (any(examsSubject.cancerL == 1)):
            leftCancer[subjectIds[i]] = 1
        else:
            leftCancer[subjectIds[i]] = 0

        if (any(examsSubject.cancerR == 1)):
            rightCancer[subjectIds[i]] = 1
                else:
            rightCancer[subjectIds[i]] = 0

        subjectData = imagesNoPatches.loc[imagesNoPatches.subjectId ==
                                          subjectIds[i], ]

        pred_LR = np.zeros(2) # contains L = 0, R = 1

        # predict L
        leftImages = subjectData.loc[subjectData.laterality == "L", ]
        if leftImages.shape[0] == 0:
            print "no Left images for this subject"
        else:
            print "num. L views: {}" .format(leftImages.shape[0])
            # contains the score for each image in the laterality
            pred_LR[0] = max(scoresAll[leftImages.index, 1])
                
        # predictR
        rightImages = subjectData.loc[subjectData.laterality == "R", ]
        if rightImages.shape[0] == 0:
            print "no Right images for this subject"
        else:
            print "num. R views: {}" .format(rightImages.shape[0])
            # contains the score for each image in the laterality
            pred_LR[1] = max(scoresAll[rightImages.index, 1])

        # Update the subject scores
        scores[subjectIds[i]] = pred_LR

    # Print the AUC scores for L and R
    scoresLR = np.array([scores[ID] for ID in sorted(scores)])
    leftC = np.array([leftCancer[ID] for ID in sorted(leftCancer)])
    rightC = np.array([rightCancer[ID] for ID in sorted(rightCancer)])
    if (any(leftC == 1)):
        aucL = roc_auc_score(leftC, scoresLR[:, 0])
        print "AUC left: {}" .format(aucL)
    else:
        print "only 1 class"
    if (any(rightC == 1)):
        aucR = roc_auc_score(rightC, scoresLR[:, 1])
        print "AUC right: {}" .format(aucR)
    else:
        print "only 1 class"


