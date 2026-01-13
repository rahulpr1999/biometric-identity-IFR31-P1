import cv2
import numpy as np

def get_ifr31_alignment(cv_img, face_landmarks):
    """
    IFR31-P1 Standardized Alignment Utility
    Transforms raw face detection landmarks into a normalized 112x112 output.
    Developed by Marrty LLC
    """
    
    # Standard ArcFace destination points for 112x112 resolution
    # These coordinates ensure the eyes and nose are perfectly positioned.
    dst_pts = np.array([
        [30.2946, 51.6963],  # Left Eye
        [71.7054, 51.6963],  # Right Eye
        [51.0000, 71.7366],  # Nose Tip
        [33.5493, 92.3655],  # Left Mouth Corner
        [68.4507, 92.3655]   # Right Mouth Corner
    ], dtype=np.float32)

    # Convert detected landmarks to float32
    src_pts = np.array(face_landmarks, dtype=np.float32)

    # Calculate the Similarity Transformation (Rotation + Scaling + Translation)
    # estimateAffinePartial2D is more stable for facial alignment than a full perspective warp.
    tform, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    if tform is None:
        return None

    # Warp the image to the standardized IFR31 dimensions
    aligned_face = cv2.warpAffine(cv_img, tform, (112, 112))
    
    return aligned_face

if __name__ == "__main__":
    print("🛡️ Marrty LLC: IFR31-P1 Alignment Module Loaded.")
