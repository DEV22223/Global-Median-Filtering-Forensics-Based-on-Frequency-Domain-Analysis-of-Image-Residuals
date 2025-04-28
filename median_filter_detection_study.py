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
