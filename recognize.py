# recognize.py
import json, os, argparse, cv2, numpy as np
from common import detect_box, compute_embedding

GALLERY = "gallery.json"
DEFAULT_THRESH = 0.42   # <-- set your preferred default threshold here

def recognize(img_path, thresh=DEFAULT_THRESH):
    """
    Returns (best_id, best_distance, is_match) using Euclidean distance.
    """
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(img_path)
    box = detect_box(bgr)
    if not box:
        return None, None, False
    q = compute_embedding(bgr, box)
    if q is None:
        return None, None, False

    if not os.path.exists(GALLERY):
        return None, None, False
    data = json.load(open(GALLERY))

    best_id, best_d = None, 1e9
    for pid, vec in data.items():
        d = np.linalg.norm(q - np.array(vec, dtype=np.float32))
        if d < best_d:
            best_id, best_d = pid, d
    return best_id, float(best_d), (best_d < thresh)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="Path to image to recognize")
    ap.add_argument("--thresh", type=float, default=DEFAULT_THRESH,
                    help=f"Euclidean threshold (default {DEFAULT_THRESH})")
    args = ap.parse_args()

    pid, dist, ok = recognize(args.img, args.thresh)
    print({"match": ok, "person_id": pid, "distance": dist})

if __name__ == "__main__":
    main()