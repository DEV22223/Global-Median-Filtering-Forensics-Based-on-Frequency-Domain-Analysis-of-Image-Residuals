"""
Image Forensics Demo: Median Filter Detection
Author: Your Name
License: MIT
"""

# %% === Setup ===
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set reproducible results
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# %% === Core Functions from Implementation ===

def center_crop(img, size):
    # ... (same as original implementation) ...

def jpeg_compress(img, quality):
    # ... (same as original implementation) ...

def median_filter(img, size):
    # ... (same as original implementation) ...

def compute_gdctf_features(img, ksize=3):
    # ... (same as original implementation) ...

# %% === Example Image Processing Pipeline ===

# Load sample image
china = load_sample_image('china.jpg')
gray_img = cv2.cvtColor(china, cv2.COLOR_RGB2GRAY)

# Processing demonstration
plt.figure(figsize=(12, 6))

# Original
plt.subplot(2, 3, 1)
plt.imshow(gray_img, cmap='gray')
plt.title('Original Image')

# Center cropped
cropped, _ = center_crop(gray_img, 256)
plt.subplot(2, 3, 2)
plt.imshow(cropped, cmap='gray')
plt.title('Center Cropped (256x256)')

# JPEG compressed
jpeg_img = jpeg_compress(cropped, 75)
plt.subplot(2, 3, 3)
plt.imshow(jpeg_img, cmap='gray')
plt.title('JPEG Q75')

# Median filtered
filtered_img = median_filter(jpeg_img, 3)
plt.subplot(2, 3, 4)
plt.imshow(filtered_img, cmap='gray')
plt.title('Median Filtered (3x3)')

# Feature extraction
features = compute_gdctf_features(jpeg_img)
plt.subplot(2, 3, 5)
plt.bar(['Mean', 'Variance', 'Skewness', 'Kurtosis'], features)
plt.title('DCT Features')

plt.tight_layout()
plt.show()

# %% === Quick Classification Demo ===

def generate_demo_data(n_samples=100, img_size=128):
    """Generate synthetic dataset for demonstration"""
    X = []
    y = []
    for _ in range(n_samples):
        # Generate random texture
        clean = np.random.randint(0, 256, (img_size, img_size), dtype=np.uint8)
        
        # Add random manipulations
        if np.random.rand() > 0.5:
            processed = median_filter(clean, 3)
            y.append(1)
        else:
            processed = clean
            y.append(0)
            
        # Extract features
        X.append(compute_gdctf_features(processed))
        
    return np.array(X), np.array(y)

# Generate demo dataset
X, y = generate_demo_data()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)

# Train classifier
clf = SVC(kernel='linear', random_state=RANDOM_STATE)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nDemo Classification Results:")
print(f" - Training samples: {len(X_train)}")
print(f" - Test samples: {len(X_test)}")
print(f" - Accuracy: {accuracy:.2%}")

# %% === Usage Instructions ===
"""
To run the full implementation:
1. Place images in designated directories
2. Run median_filter_detection_study.py
3. Results saved to 'median_filtering_results.csv'

Required dependencies:
!pip install opencv-python scikit-learn matplotlib numpy pandas scikit-image tqdm
"""

# %% === End of Demo ===
