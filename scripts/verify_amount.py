# scripts/verify_amount.py
import os
import cv2
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# -----------------------------
# Load EasyOCR
# -----------------------------
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    print("✅ EasyOCR loaded")
except Exception as e:
    print(f"⚠️ EasyOCR error: {e}")
    reader = None

# -----------------------------
# Load CNN Model
# -----------------------------
# Assumes you have trained a CNN on handwritten digits (0-9) like MNIST
try:
    cnn_model = load_model("amount.keras")
    print("✅ CNN model loaded")
except Exception as e:
    print(f"⚠️ CNN model error: {e}")
    cnn_model = None

# -----------------------------
# Directories
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
TEMP_DIR = os.path.join(PARENT_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# -----------------------------
# Helper: CNN Digit Prediction
# -----------------------------
def predict_amount_with_cnn(img_path):
    """
    Predict handwritten amount using CNN (digit by digit)
    """
    if cnn_model is None:
        return 0, 0.0

    # Load image in grayscale
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0, 0.0

    # Threshold and invert
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours (digits)
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits = []
    
    for cnt in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5:
            continue  # ignore small noise
        digit_img = img[y:y+h, x:x+w]
        # Resize to 28x28
        digit_img = cv2.resize(digit_img, (28, 28))
        digit_img = digit_img.astype("float32") / 255.0
        digit_img = np.expand_dims(digit_img, axis=-1)
        digit_img = np.expand_dims(digit_img, axis=0)
        pred = cnn_model.predict(digit_img, verbose=0)
        digit = np.argmax(pred)
        digits.append(str(digit))

    if digits:
        amount_number = int("".join(digits))
        return amount_number, 0.9  # confidence placeholder
    else:
        return 0, 0.0

# -----------------------------
# Main OCR Function
# -----------------------------
def predict_amount(img_path):
    """
    Combined OCR + CNN extraction
    Returns: (text, number, confidence)
    """
    print("\n" + "="*60)
    print("EXTRACTING AMOUNT WITH OCR + CNN")
    print("="*60)
    
    if reader is None:
        print("❌ OCR not available")
        return "0", 0, 0.0
    
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "0", 0, 0.0
        
        h, w = img.shape
        if w < 800:
            scale = 800 / w
            img = cv2.resize(img, (int(w*scale), int(h*scale)), cv2.INTER_CUBIC)
        
        # Threshold for OCR
        _, img_thr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Save debug
        debug_path = os.path.join(TEMP_DIR, "amount_debug.jpg")
        cv2.imwrite(debug_path, img_thr)

        # -----------------------------
        # Run OCR
        # -----------------------------
        results = reader.readtext(debug_path, detail=1, allowlist='0123456789,.')

        all_texts = []
        all_numbers = []
        confidences = []

        for (_, text, conf) in results:
            text = text.strip()
            if text and conf > 0.1:
                all_texts.append(text)
                confidences.append(conf)
                digits = re.sub(r'[^\d]', '', text)
                if digits:
                    try:
                        all_numbers.append(int(digits))
                    except:
                        pass

        # Pick best OCR result
        if all_numbers:
            best_idx = np.argmax([len(str(num)) for num in all_numbers])
            ocr_text = all_texts[best_idx]
            ocr_number = all_numbers[best_idx]
            ocr_conf = np.mean(confidences)
        else:
            ocr_text, ocr_number, ocr_conf = "0", 0, 0.0

        # -----------------------------
        # Run CNN if OCR confidence low
        # -----------------------------
        if ocr_conf < 0.7 and cnn_model is not None:
            cnn_number, cnn_conf = predict_amount_with_cnn(img_path)
            if cnn_conf > ocr_conf:
                print(f"⚠️ OCR low confidence ({ocr_conf:.2f}), using CNN result")
                return str(cnn_number), cnn_number, cnn_conf

        print(f"\n✅ EXTRACTED AMOUNT:")
        print(f"   OCR Text: '{ocr_text}'")
        print(f"   Number: {ocr_number}")
        print(f"   Confidence: {ocr_conf:.2f}")
        print("="*60 + "\n")

        return ocr_text, ocr_number, ocr_conf

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return "0", 0, 0.0

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    test_image = os.path.join(TEMP_DIR, "sample_cheque_amount.jpg")
    text, number, conf = predict_amount(test_image)
    print(f"Final Result -> Text: {text}, Number: {number}, Confidence: {conf:.2f}")
