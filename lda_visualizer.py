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
