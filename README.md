# Global-Median-Filtering-Forensics-Based-on-Frequency-Domain-Analysis-of-Image-Residuals
Detecting median filtering in images is crucial for digital forensics Existing methods struggle with low-resolution, heavily compressed images Proposes a new 4D feature set based on frequency domain analysis of Median Filtered Residuals (MFR)
# Median Filtering Forensics - Implementation

This repository contains the implementation of the paper:  
**"A Simplistic Global Median Filtering Forensics Based on Frequency Domain Analysis of Image Residuals"**  
by Abhinav Gupta and Divya Singhal.

## 📝 Paper Overview

The paper proposes a novel 4-dimensional feature set called **GDCTF** (Global DCT Features) for detecting median filtering in digital images, particularly effective for:
- Low-resolution images (128×128, 64×64, 32×32)
- JPEG compressed images (quality factors 30-90)
- Robust against anti-forensic attacks
- Outperforms both handcrafted and CNN-based state-of-the-art methods

Key contributions:
1. Frequency domain analysis of Median Filtered Residuals (MFR)
2. Small feature dimensions (4 features) with superior performance
3. Resolution-invariant detection capability

## 🛠 Implementation Details

### 🔍 Core Components

1. **Feature Extraction** (`compute_gdctf_features()`):
   - Computes 2D Global DCT of image residuals
   - Extracts 4 statistical moments: Mean, Variance, Skewness, Kurtosis

2. **Image Processing Pipeline**:
   - Center cropping (`center_crop()`)
   - JPEG compression (`jpeg_compress()`)
   - Median filtering (`median_filter()`)
   - Anti-forensic manipulation handling

3. **Classification Framework**:
   - Linear SVM classifier
   - 4-fold cross validation
   - StandardScaler for feature normalization

### 📊 Key Features

- **Multi-database support**: UCID, BOWS2, BOSSBase, RAISE, NRCS
- **Multiple resolutions**: 128×128, 64×64, 32×32
- **Quality factors**: Uncompressed, Q90, Q70, Q50, Q30
- **Filter sizes**: 3×3 and 5×5 median filters
- **Anti-forensic robustness**: Implements Fan et al.'s anti-forensic method

## 🏗 Project Structure

```
median-forensics/
├── data_processing/          # Image loading and preprocessing
│   ├── center_crop.py
│   ├── jpeg_compress.py
│   └── median_filter.py
│
├── feature_extraction/       # GDCTF feature computation
│   ├── gdctf.py
│   └── residuals.py
│
├── classification/           # Machine learning models
│   ├── svm_classifier.py
│   └── evaluation.py
│
├── visualization/            # Plotting and results visualization
│   ├── histograms.py
│   ├── lda_projections.py
│   └── roc_curves.py
│
├── datasets/                 # Sample datasets (links to download)
│
└── main.py                   # Main pipeline
```

## 🚀 Getting Started

### Prerequisites
- Python 3.6+
- OpenCV
- scikit-learn
- scikit-image
- NumPy
- Matplotlib
- tqdm

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from main import run_experiment

# Run full experimental pipeline
results = run_experiment(
    dataset_path="path/to/images",
    crop_size=128,
    quality_factors=[90, 70, 50, 30],
    filter_sizes=[3, 5]
)
```

## 📈 Results Reproduction

The implementation includes scripts to reproduce all key figures from the paper:

1. **Figure 1**: Image and residual visualization
   ```bash
   python visualization/image_comparison.py
   ```

2. **Figures 2-3**: GDCT coefficient histograms
   ```bash
   python visualization/histograms.py
   ```

3. **Figure 4**: Feature ratio analysis
   ```bash
   python visualization/feature_ratios.py
   ```

4. **Figure 6**: LDA projections
   ```bash
   python visualization/lda_projections.py
   ```

5. **Figure 7**: ROC curves
   ```bash
   python visualization/roc_curves.py
   ```

## 📚 Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{gupta2019simplistic,
  title={A Simplistic Global Median Filtering Forensics Based on Frequency Domain Analysis of Image Residuals},
  author={Gupta, Abhinav and Singhal, Divya},
  journal={ACM Transactions on Multimedia Computing, Communications, and Applications},
  volume={15},
  number={3},
  pages={1--23},
  year={2019},
  publisher={ACM}
}
```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
