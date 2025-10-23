# enroll.py
import json, os, argparse, cv2
from common import detect_box, compute_embedding

GALLERY = "gallery.json"  # person_id -> embedding list

def save_embedding(person_id, emb):
    data = {}
    if os.path.exists(GALLERY):
        with open(GALLERY, "r") as f:
            data = json.load(f)
    data[person_id] = [float(x) for x in emb]
    with open(GALLERY, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Enrolled {person_id}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="Employee/person ID")
    ap.add_argument("--img", required=True, help="Path to a face image")
    args = ap.parse_args()

    bgr = cv2.imread(args.img)
    if bgr is None:
        raise FileNotFoundError(args.img)
    box = detect_box(bgr)
    if not box:
        raise RuntimeError("No face detected. Try a clearer, frontal photo.")
    emb = compute_embedding(bgr, box)
    if emb is None:
        raise RuntimeError("Could not compute embedding.")
    save_embedding(args.id, emb)

if __name__ == "__main__":
    main()