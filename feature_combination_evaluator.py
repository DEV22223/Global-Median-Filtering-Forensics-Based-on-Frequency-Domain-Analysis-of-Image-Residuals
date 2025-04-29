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
