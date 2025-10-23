# common.py
import cv2, numpy as np, face_recognition

# OpenCV DNN face detector (ResNet-SSD 300x300)
PROTO = "deploy.prototxt"
MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
_net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

def detect_box(bgr, conf_thresh=0.6):
    """
    Returns first face box (x1,y1,x2,y2) or None.
    """
    h, w = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    _net.setInput(blob)
    dets = _net.forward()
    for i in range(dets.shape[2]):
        conf = dets[0, 0, i, 2]
        if conf >= conf_thresh:
            x1 = int(dets[0, 0, i, 3] * w)
            y1 = int(dets[0, 0, i, 4] * h)
            x2 = int(dets[0, 0, i, 5] * w)
            y2 = int(dets[0, 0, i, 6] * h)
            return (x1, y1, x2, y2)
    return None

def to_fr_box(box):
    # Convert (x1,y1,x2,y2) to face_recognition format: (top, right, bottom, left)
    x1, y1, x2, y2 = box
    return (y1, x2, y2, x1)

def compute_embedding(bgr, box=None, num_jitters=1):
    """
    Returns a 128-D face embedding (numpy array) or None.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    locs = [to_fr_box(box)] if box else face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(
        rgb, known_face_locations=locs, num_jitters=num_jitters
    )
    return encs[0] if len(encs) else None