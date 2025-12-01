# scripts/segmentation.py
import cv2
import os

def segment_cheque(image_path, out_dir="temp"):
    """
    Exact coordinates tuned for your specific cheque image (based on highlights).
    - Amount: "5,000" box in top right corner
    - Signature: Main signature in center-left area
    - Padding: +20px white for clean model input.
    """
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    h, w = img.shape[:2]
    print(f"Debug: Image size {w}x{h}")
    
    # Reference size (based on your image aspect ratio ~1280x560)
    ref_w, ref_h = 1280, 560
    scale_w = w / ref_w
    scale_h = h / ref_h
    
    # Amount box coordinates (top right, blue rectangle around "5,000")
    amt_x1 = int(865 * scale_w)   # Left edge of amount box
    amt_y1 = int(175 * scale_h)   # Top edge
    amt_x2 = int(1195 * scale_w)  # Right edge
    amt_y2 = int(245 * scale_h)   # Bottom edge
    amt_crop = img[amt_y1:amt_y2, amt_x1:amt_x2]
    
    # Signature coordinates (center-left, blue rectangle around signature)
    sig_x1 = int(575 * scale_w)   # Left edge of signature box
    sig_y1 = int(255 * scale_h)   # Top edge
    sig_x2 = int(870 * scale_w)   # Right edge
    sig_y2 = int(420 * scale_h)   # Bottom edge
    sig_crop = img[sig_y1:sig_y2, sig_x1:sig_x2]
    
    # Add padding (white border)
    def add_padding(crop, pad=20):
        return cv2.copyMakeBorder(crop, pad, pad, pad, pad, 
                                  cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    sig_path = os.path.join(out_dir, "cheque_sign.jpg")
    amt_path = os.path.join(out_dir, "cheque_amount.jpg")
    
    cv2.imwrite(sig_path, add_padding(sig_crop))
    cv2.imwrite(amt_path, add_padding(amt_crop))
    
    print(f"Signature saved: {sig_path}")
    print(f"Amount saved: {amt_path}")
    
    return sig_path, amt_path