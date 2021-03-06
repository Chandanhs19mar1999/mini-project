from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import h5py
import numpy as np
import os
import cv2
import pickle
import warnings
import base64
import io
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split,cross_val_score
from PIL import Image
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from django.core.files.storage import FileSystemStorage
import mahotas
@csrf_exempt
def index(request):
    categories = ['Healthy', 'bacterialSpot', 'lateblight', 'septoriaLeafSpot', 'tomato_mosaic', 'yellowcurved']
    rfc = open("ml_models/rfc_classifier.pickle","rb")
    #print(request.body,"\n\n\n")
    img=request.FILES['image']
    img=Image.open(img)
    # print(request)
    # print()
    # print(img)
    # print(type(img))
    clf = pickle.load(rfc)
    fixed_size = tuple((500, 500))
    #bins=8
    #gray = cv2.cvtColor(imgUMat, cv2.COLOR_BGR2GRAY)
    #image = cv2.UMat(img)
    image = cv2.resize(np.float32(img), fixed_size)
    image= image.astype('uint8')
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)
    globalfeature = np.hstack([fv_histogram,fv_hu_moments, fv_haralick])
    # scale features in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    globalfeature=globalfeature.reshape(4,133)
    rescaled_feature = scaler.fit_transform(globalfeature)
    testimage=np.array(rescaled_feature)
    testimage= testimage.flatten()
    testimage=testimage.reshape(1,-1)
    prediction = clf.predict(testimage)
    res=categories[prediction[0]]
    print(res)
    print(request)
    #here
    return HttpResponse(res)
# Create your views here.

def fd_hu_moments(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature = cv2.HuMoments(cv2.moments(image))
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        return feature

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

def fd_histogram(image, mask=None):
    bins=8
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def getImage(b64str):
    imgdata = base64.b64decode(str(b64str))
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(cv2.UMat(image), cv2.COLOR_BGR2RGB)