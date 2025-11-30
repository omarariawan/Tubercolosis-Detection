import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog

# --- CONFIGURATION ---
DATASET_PATH = '.'
IMG_SIZE = (512, 512)
MODEL_FILENAME = 'svm_tb_model.pkl'

def preprocess_image(image_path):
    """
    Reads image, converts to grayscale, applies CLAHE and Median Filter.
    """
    img = cv2.imread(image_path)
    if img is None: 
        return None
    
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize
    gray = cv2.resize(gray, IMG_SIZE)
    
    # 3. CLAHE (Contrast Enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 4. Noise Reduction
    filtered = cv2.medianBlur(enhanced, 3)
    
    return filtered

def segment_lung(image):
    """
    Isolates lung area using Otsu Thresholding.
    """
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    segmented = cv2.bitwise_and(image, image, mask=opening)
    return segmented

def extract_features(image):
    """
    Extracts GLCM, LBP, and HOG features.
    """
    feature_vector = []
    
    # --- 1. GLCM ---
    distances = [1]
    # Angles in radians: 0, 45, 90, 135
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    props = ['contrast', 'correlation', 'energy', 'homogeneity']
    for prop in props:
        # flattens the 4 angles into a 1D list
        feature_vector.extend(graycoprops(glcm, prop).flatten())

    # --- 2. LBP ---
    radius = 2
    n_points = 16
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    # Use histogram of LBP as the feature
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7) # Normalize
    feature_vector.extend(lbp_hist)

    # --- 3. HOG ---
    hog_feats = hog(image, orientations=9, pixels_per_cell=(64, 64), 
                    cells_per_block=(2, 2), visualize=False, feature_vector=True)
    feature_vector.extend(hog_feats)
    
    return np.array(feature_vector).reshape(1, -1)  # Reshape for model.predict()

def train():
    print("--- Starting Training Pipeline ---")
    data = []
    labels = []
    
    # 0 = Normal, 1 = Tuberculosis
    categories = {'Normal': 0, 'Tuberculosis': 1}
    
    for category, label in categories.items():
        folder_path = os.path.join(DATASET_PATH, category)
        print(f"Processing folder: {category}...")
        
        # Process ALL images from the folder
        all_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        image_files = all_files
        print(f"  Total {category} images found: {len(image_files)}")
        
        for idx, img_name in enumerate(image_files, 1):
            img_path = os.path.join(folder_path, img_name)
            
            try:
                # 1. Preprocess
                processed = preprocess_image(img_path)
                if processed is None: 
                    print(f"  Skipped {img_name} - could not process")
                    continue
                
                # 2. Segment
                segmented = segment_lung(processed)
                
                # 3. Extract Features
                feats = extract_features(segmented)
                
                data.append(feats[0])  # Extract the 1D array from (1, n_features)
                labels.append(label)
                
                if idx % 500 == 0:
                    print(f"  Processed {idx}/{len(image_files)} {category} images...")
            except Exception as e:
                print(f"  Error processing {img_name}: {str(e)}")

    print(f"Feature extraction complete. Total images: {len(data)}")
    
    X = np.array(data)
    y = np.array(labels)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize Data (Crucial for SVM)
    print("Normalizing data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) # Use same scaler for test data
    
    # Train SVM
    print("Training SVM Model (RBF Kernel)...")
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True) #
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTraining Finished!")
    print(f"Accuracy: {acc*100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Save Model AND Scaler
    # We save the scaler too because new images must be scaled exactly like the training data
    joblib.dump(model, MODEL_FILENAME)
    joblib.dump(scaler, 'scaler.pkl')
    
    print(f"Model saved as '{MODEL_FILENAME}'")
    print(f"Scaler saved as 'scaler.pkl'")

if __name__ == "__main__":
    train()