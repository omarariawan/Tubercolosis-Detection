# feature_extractor.py
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog

# Image configuration
IMG_SIZE = (512, 512)

def preprocess_image(image_file):
    """
    Reads an image file buffer (from Flask), converts to grayscale,
    applies CLAHE and Median Filter.
    """
    # Convert file stream to numpy array
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Image could not be decoded.")

    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize
    gray = cv2.resize(gray, IMG_SIZE)
    
    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 4. Median Filter
    filtered = cv2.medianBlur(enhanced, 3)
    
    return filtered

def segment_lung(image):
    """Isolates lung regions using Otsu thresholding."""
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    segmented = cv2.bitwise_and(image, image, mask=opening)
    return segmented

def extract_features(image):
    """Extracts GLCM, LBP, and HOG features."""
    feature_vector = []
    
    try:
        # 1. GLCM
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        props = ['contrast', 'correlation', 'energy', 'homogeneity']
        for prop in props:
            feature_vector.extend(graycoprops(glcm, prop).flatten())

        # 2. LBP
        radius = 2
        n_points = 16
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-7)
        feature_vector.extend(lbp_hist)

        # 3. HOG
        hog_feats = hog(image, orientations=9, pixels_per_cell=(64, 64), 
                        cells_per_block=(2, 2), visualize=False, feature_vector=True)
        feature_vector.extend(hog_feats)
        
        return np.array(feature_vector).reshape(1, -1) # Reshape for single prediction
    except Exception as e:
        raise ValueError(f"Feature extraction failed: {str(e)}")