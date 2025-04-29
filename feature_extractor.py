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
