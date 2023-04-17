import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from skimage.feature import peak_local_max


def NMS(
    tableHit,
    scoreThreshold=0.5,
    N_object=float('inf'),
    maxOverlap=0.5,
    ):
    """
    Perform Non-Maxima Supression (NMS).
    compares the hits after maxima detection, and removes detections overlapping above the maxOverlap  
   
    Parameters
    - tableHit : List of potential detections 
                        
    - scoreThreshold : Float, used to remove low score detections, the detections with score below the threshold are discarded.
                       
    - N_object      : number of best detections to return
    
    - maxOverlap    : the maximal overlap (IoU: Intersection over Union of the areas) authorised between 2 bounding boxes. 
                     
    
    Returns
    -------
    Panda DataFrame with best detection after NMS, it contains max N detections
    """

    listBoxes = tableHit['BBox'].to_list()
    listScores = tableHit['Score'].to_list()

    indexes = cv2.dnn.NMSBoxes(listBoxes, listScores, scoreThreshold,
                               maxOverlap)

    indexes = indexes[:N_object]

    outTable = tableHit.iloc[indexes]

    return outTable


def _multi_compute(
    template,
    image,
    method,
    N_object,
    score_threshold,
    ):
    """
    Find all possible template locations satisfying the score threshold provided a template to search and an image.
    Add the hits in the list of hits.
    
    Parameters
    ----------
    - template : template used to find similar images within the image

    - image  : image in which to perform the search
    
    - method : method used for search
    
    - N_object: expected number of objects in the image
    
    - score_threshold: find local maxima above the score_threshold
    
    - maxOverlap: maximal value for the ratio of the Intersection Over Union (IoU) area between a pair of bounding boxes    
    

    Returns
    -------
    Panda DataFrame with detection
    """

    # store hits in this list

    listHit = []

    # search using specified algorithm

    corrMap = cv2.matchTemplate(image, template, method)

    # Find possible location of the object

    peaks = peak_local_max(corrMap, threshold_abs=score_threshold,
                           exclude_border=False).tolist()

    # add width and height as original template

    (height, width) = template.shape[0:2]

    for peak in peaks:

        # append to list of potential hit before Non maxima suppression
        # no need to lock the list, append is thread-safe

        listHit.append({'TemplateName': 'template',
                       'BBox': (int(peak[1]), int(peak[0]), width,
                       height), 'Score': corrMap[tuple(peak)]})  # empty df with correct column header
    return pd.DataFrame(listHit)


def matchTemplates(
    template,
    image,
    method=cv2.TM_CCOEFF_NORMED,
    N_object=8,
    score_threshold=0.5,
    maxOverlap=0.25,
    ):
    """
    Function to search for template and return results that have overlap less than threshold
    
    Parameters
    ----------
    - template : template used to find similar images within the image
    
    - image  : image in which to perform the search
    
    - method : method used for search
    
    - N_object: expected number of objects in the image
    
    - score_threshold: find local maxima above the score_threshold
    
    - maxOverlap: maximal value for the ratio of the Intersection Over Union (IoU) area between a pair of bounding boxes
    
    
    Returns
    -------
    Pandas DataFrame with 1 row per hit and column after applying NMS
    """

    tableHit = _multi_compute(template, image, method, N_object,
                              score_threshold)

    return NMS(tableHit, score_threshold, N_object, maxOverlap)
