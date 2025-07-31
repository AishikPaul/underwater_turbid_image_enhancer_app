import cv2
import numpy as np

def apply_bilateral_filter(img):
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

def get_dark_channel(img, window_size=15):
    min_img = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_img, kernel)
    return dark_channel

def get_atmosphere(img, dark_channel, top_percent=0.001):
    flat_img = img.reshape(-1, 3)
    flat_dark = dark_channel.ravel()
    num_pixels = int(max(flat_dark.shape[0] * top_percent, 1))
    indices = np.argpartition(-flat_dark, num_pixels)[:num_pixels]
    A = np.max(flat_img[indices], axis=0)
    return A

def get_transmission(img, A, omega=0.95, window_size=15):
    normed = img / A
    transmission = 1 - omega * get_dark_channel(normed, window_size)
    return transmission

def guided_filter(I, p, r, eps):
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    return mean_a * I + mean_b

def recover_scene_radiance(img, A, transmission, t0=0.1):
    transmission = np.clip(transmission, t0, 1)
    J = (img - A) / transmission[..., np.newaxis] + A
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J

def dehaze_image(img):
    dark_channel = get_dark_channel(img)
    A = get_atmosphere(img, dark_channel)
    transmission_est = get_transmission(img, A)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    transmission_refined = guided_filter(gray, transmission_est, r=40, eps=1e-3)
    dehazed = recover_scene_radiance(img.astype(np.float64), A, transmission_refined)
    return dehazed

def white_balance(img):
    result = img.copy()
    for i in range(3):
        hist, _ = np.histogram(img[..., i].flatten(), 256, [0,256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        result[..., i] = cdf[img[..., i]]
    return result

def apply_clahe_on_v(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe.apply(v)
    hsv[:, :, 2] = v
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def enhance_image(image_bytes):
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bilateral = apply_bilateral_filter(img)
    dehazed = dehaze_image(bilateral)
    white_balanced = white_balance(dehazed)
    enhanced_rgb = apply_clahe_on_v(white_balanced)
    enhanced_gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)

    # Save outputs
    cv2.imwrite("static/output_rgb.png", cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite("static/output_gray.png", enhanced_gray)

    return img, enhanced_rgb, enhanced_gray
