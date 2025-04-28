#!pip install pandas scikit-image
import os
import cv2
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.util import random_noise
from skimage.transform import rescale
import pandas as pd
import seaborn as sns
from itertools import combinations

