"def center_crop(img, size):
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
