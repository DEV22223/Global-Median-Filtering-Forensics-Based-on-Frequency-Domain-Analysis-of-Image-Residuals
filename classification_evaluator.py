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
