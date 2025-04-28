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

# === Dataset directories ===
DATASET_DIRS = [
    "/content/drive/MyDrive/UCID1338",
    "/content/drive/MyDrive/boss_256_0.4_test",
    "/content/drive/MyDrive/boss_256_0.4",
    "/content/drive/MyDrive/wow0.2"
]

# === Experiment settings ===
CROP_SIZES = [128, 64, 32]
QUALITY_FACTORS = [90, 70, 50, 30, None]  # None = uncompressed
FILTER_SIZES = [3, 5]
RANDOM_STATE = 42
MAX_SAMPLES = 200  # For demo; remove or increase for full data

# === Visualization settings ===
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

# Helper functions ==============================================

def center_crop(img, size):
    h, w = img.shape[:2]
    if h < size or w < size:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA), True
    top = (h - size) // 2
    left = (w - size) // 2
    return img[top:top+size, left:left+size], False


def jpeg_compress(img, quality):
    if quality is None:
        return img
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def median_filter(img, size):
    return cv2.medianBlur(img, size)


def compute_gdctf_features(img, ksize=3):
    med = median_filter(img, ksize)
    resid = med.astype(np.float32) - img.astype(np.float32)
    dct_coeff = cv2.dct(resid)
    coeffs = dct_coeff.flatten()
    mu = coeffs.mean()
    var = coeffs.var()
    sk = skew(coeffs) if var > 0 else 0.0
    ku = kurtosis(coeffs, fisher=False) if var > 0 else 0.0
    return np.nan_to_num(np.array([mu, var, sk, ku], dtype=np.float32))


def apply_anti_forensic(img, method='fan2015'):
    if method == 'fan2015':
        noise = np.random.normal(0, 0.5, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def apply_other_manipulations(img, manipulation):
    if manipulation == 'avg':
        return cv2.blur(img, (3,3))
    elif manipulation == 'gau':
        return cv2.GaussianBlur(img, (3,3), 0.5)
    elif manipulation == 'rescale':
        scaled = rescale(img, 1.5, anti_aliasing=True)
        h, w = img.shape
        return center_crop((scaled*255).astype(np.uint8), h)[0]
    return img

# Feature extraction ============================================

def extract_features(dataset_dir, crop_size, quality, filter_size=None,
                    manipulation=None, anti_forensic=False, max_samples=None):
    feats = []
    for fname in tqdm(os.listdir(dataset_dir)[:max_samples], desc=f'Extracting {crop_size} Q{quality}'):
        if not fname.lower().endswith(('.jpg','.jpeg','.png','.tif','.bmp','.pgm')):
            continue
        img = cv2.imread(os.path.join(dataset_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img_c, _ = center_crop(img, crop_size)
        img_q = jpeg_compress(img_c, quality)
        if manipulation:
            img_q = apply_other_manipulations(img_q, manipulation)
        elif filter_size:
            img_q = median_filter(img_q, filter_size)
            if anti_forensic:
                img_q = apply_anti_forensic(img_q)
        feats.append(compute_gdctf_features(img_q))
    return np.array(feats)

# Visualization functions =======================================
# [plot_image_comparison, plot_gdct_histograms, plot_feature_ratios, plot_lda_projections, plot_roc_curve, plot_confusion_matrix]

# NEW: Feature combination analysis =============================

def evaluate_feature_combinations(X, y, title):
    comb_indices = [[0,2],[0,1],[0,3],[0,1,2],[0,1,3],[0,2,3],[0,1,2,3]]
    comb_labels = ['μ+Sk','μ+σ²','μ+Ku','μ+σ²+Sk','μ+σ²+Ku','μ+Sk+Ku','All']
    pe_results = []
    for inds in comb_indices:
        Xs = StandardScaler().fit_transform(X[:, inds])
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
        pe_list = []
        for tr, te in skf.split(Xs, y):
            clf = SVC(kernel='linear', random_state=RANDOM_STATE)
            clf.fit(Xs[tr], y[tr])
            pe_list.append(1 - accuracy_score(y[te], clf.predict(Xs[te])))
        pe_results.append(np.mean(pe_list)*100)
    plt.figure(figsize=(10,6))
    bars=plt.bar(comb_labels, pe_results, color='skyblue')
    for bar,h in zip(bars,pe_results): plt.text(bar.get_x()+bar.get_width()/2, h, f'{h:.1f}%', ha='center', va='bottom')
    plt.title(f'Feature Combination Analysis: {title}')
    plt.ylabel('Probability of Error (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(); plt.show()

# NEW: Combined LDA for MF3 vs MF5 ============================

def plot_combined_lda(dataset_dir, crop, quality, max_samples):
    Xo = extract_features(dataset_dir, crop, quality, None, None, False, max_samples)
    X3 = extract_features(dataset_dir, crop, quality, 3, None, False, max_samples)
    X5 = extract_features(dataset_dir, crop, quality, 5, None, False, max_samples)
    X = np.vstack((Xo,X3,X5))
    y = np.hstack((np.zeros(len(Xo)), np.ones(len(X3)), np.full(len(X5), 2)))
    lda = LinearDiscriminantAnalysis(n_components=2)
    proj = lda.fit_transform(StandardScaler().fit_transform(X), y)
    plt.figure(figsize=(8,6))
    for cls, m, lbl, col in zip([0,1,2], ['o','s','^'], ['Original','MF3','MF5'], ['navy','darkorange','green']):
        plt.scatter(proj[y==cls,0], proj[y==cls,1], alpha=0.6, marker=m, color=col, label=lbl)
    plt.title(f'LDA Projection (Orig vs MF3 vs MF5) Q={quality}, {crop}×{crop}')
    plt.xlabel('LD1'); plt.ylabel('LD2'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(); plt.show()

# Classification & evaluation ===================================

def evaluate_classification(X, y, title, plot=True):
    Xs = StandardScaler().fit_transform(X)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    pe_list=[]; y_scores=np.zeros(len(y)); y_preds=np.zeros(len(y))
    for tr, te in skf.split(Xs, y):
        clf = SVC(kernel='linear', probability=True, random_state=RANDOM_STATE)
        clf.fit(Xs[tr], y[tr])
        yp = clf.predict(Xs[te])
        pe_list.append(1-accuracy_score(y[te], yp))
        y_preds[te]=yp; y_scores[te]=clf.predict_proba(Xs[te])[:,1]
    Pe=np.mean(pe_list)*100
    print(f"{title} - Probability of Error (Pe): {Pe:.2f}%")
    if plot:
        plot_roc_curve(y, y_scores, title)
        plot_confusion_matrix(y, y_preds, title)
    return Pe

# Figures generator ============================================

def generate_paper_figures():
    img = cv2.imread(os.path.join(DATASET_DIRS[0], os.listdir(DATASET_DIRS[0])[0]), cv2.IMREAD_GRAYSCALE)
    plot_image_comparison(img, 128, 90)
    Xo = extract_features(DATASET_DIRS[0],128,90,None,None,False,50)
    X3 = extract_features(DATASET_DIRS[0],128,90,3,None,False,50)
    X5 = extract_features(DATASET_DIRS[0],128,90,5,None,False,50)
    plot_gdct_histograms(Xo,X3,X5,90,128)
    Xo30=extract_features(DATASET_DIRS[0],128,30,None,None,False,50)
    X330=extract_features(DATASET_DIRS[0],128,30,3,None,False,50)
    X530=extract_features(DATASET_DIRS[0],128,30,5,None,False,50)
    plot_gdct_histograms(Xo30,X330,X530,30,128)
    plot_feature_ratios(Xo,X3,X5,90,128)
    plot_feature_ratios(Xo30,X330,X530,30,128)
    X = np.vstack([Xo,X3]); y = np.hstack([np.zeros(len(Xo)),np.ones(len(X3))])
    plot_lda_projections(StandardScaler().fit_transform(X), y, "LDA: Orig vs MF3 Q=90 128×128")

# Main pipeline ===============================================

def main():
    print("Generating paper figures...")
    generate_paper_figures()
    print("\nRunning main experiments...")
    results=[]
    for crop in CROP_SIZES:
        for q in QUALITY_FACTORS:
            print(f"\n=== Processing {crop}×{crop} Q={q} ===")
            Xo = extract_features(DATASET_DIRS[0],crop,q,None,None,False,MAX_SAMPLES)
            X3 = extract_features(DATASET_DIRS[0],crop,q,3,None,False,MAX_SAMPLES)
            X5 = extract_features(DATASET_DIRS[0],crop,q,5,None,False,MAX_SAMPLES)
            if len(Xo)==0 or len(X3)==0 or len(X5)==0:
                print(f"Skipping {crop}×{crop} Q={q} - insufficient samples")
                continue
            # Feature combination (Fig.5)
            Xc = np.vstack([Xo,X3]); yc = np.hstack([np.zeros(len(Xo)),np.ones(len(X3))])
            evaluate_feature_combinations(Xc, yc, f"{crop}×{crop} Q={q}")
            # Classification & summary
            for filt, Xf in [(3,X3),(5,X5)]:
                X = np.vstack([Xo,Xf]); y = np.hstack([np.zeros(len(Xo)),np.ones(len(Xf))])
                pe = evaluate_classification(X,y,f"{crop}×{crop} Q={q} MF{filt}")
                results.append({'Resolution':f"{crop}×{crop}",'Quality':q or 'UnC','Filter':f"MF{filt}",'Pe':pe})
            # Combined LDA
            print(f"Generating combined LDA for {crop}×{crop} Q={q}...")
            plot_combined_lda(DATASET_DIRS[0], crop, q, MAX_SAMPLES)
    # Summary
    df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(df.to_string(index=False))
    df.to_csv('median_filtering_results.csv', index=False)
    print("Results saved to 'median_filtering_results.csv'")

if __name__ == '__main__':
    main()
