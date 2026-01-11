import os
import numpy as np
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import math
import gc 
import urllib.request # <--- MỚI: Thư viện để tải ảnh từ URL

# --- IMPORT CUSTOM MODULES ---
import facenet
import align.detect_face

app = Flask(__name__)

# ================= CONFIG =================
SERVICE_ACCOUNT_FILE = "./service-account.json"
FACENET_MODEL_PATH = './Models/20180402-114759.pb'
COLLECTION_FACES = "FaceEmbeddings" 
LOG_COLLECTION = "AccessLogs"       

DISTANCE_THRESHOLD = 1.0 

# ================= FIREBASE SETUP =================
db = None
try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Connected to Firebase Firestore")
except Exception as e:
    print("❌ Failed to connect to Firebase:", e)

# ================= AI MODEL LOADING =================
print("⏳ Loading Facenet Model...")
sess = tf.Session()
with sess.as_default():
    facenet.load_model(FACENET_MODEL_PATH)
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    
    # Load MTCNN
    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
print("🚀 Server Ready!")

# ================= HELPER FUNCTIONS =================
def url_to_image(url):
    """MỚI: Tải ảnh từ URL và chuyển thành OpenCV Image"""
    try:
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print("Lỗi tải ảnh từ URL:", e)
        return None

def get_embedding(frame):
    """Hàm trích xuất vector đặc trưng"""
    global sess, pnet, rnet, onet
    
    # --- [ĐOẠN CODE MỚI THÊM VÀO] ---
    # Mục đích: Giảm kích thước ảnh xuống dưới 640px trước khi đưa vào AI.
    # Việc này giúp giảm RAM từ 4GB xuống chỉ còn ~200MB, tránh lỗi sập server.
    height, width = frame.shape[:2]
    max_dim = 640  # Kích thước tối đa cho phép
    
    if width > max_dim or height > max_dim:
        # Tính tỉ lệ thu nhỏ
        scale = max_dim / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        # Thực hiện resize
        frame = cv2.resize(frame, (new_width, new_height))
    # --- [HẾT ĐOẠN CODE MỚI] ---
    
    # 1. Detect Face (Đoạn này giữ nguyên như cũ)
    bounding_boxes, _ = align.detect_face.detect_face(frame, 20, pnet, rnet, onet, [0.65, 0.75, 0.75], 0.709)
    if bounding_boxes.shape[0] == 0:
        return None 
        
    # Lấy mặt to nhất
    det = bounding_boxes[:, 0:4]
    img_size = np.asarray(frame.shape)[0:2]
    
    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
    img_center = img_size / 2
    offsets = np.vstack([ (det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0] ])
    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
    index = np.argmax(bounding_box_size - offset_dist_squared * 2.0) 
    
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[index, 0], 0)
    bb[1] = np.maximum(det[index, 1], 0)
    bb[2] = np.minimum(det[index, 2], img_size[1])
    bb[3] = np.minimum(det[index, 3], img_size[0])
    
    cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
    
    # 2. Preprocess & Embedding
    scaled = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_CUBIC)
    scaled = facenet.prewhiten(scaled)
    scaled_reshape = scaled.reshape(-1, 160, 160, 3)
    
    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
    emb_array = sess.run(embeddings, feed_dict=feed_dict)
    
    return emb_array[0]
def load_known_faces():
    """Tải toàn bộ khuôn mặt đã đăng ký từ Firestore"""
    known_faces = []
    if db:
        docs = db.collection(COLLECTION_FACES).stream()
        for doc in docs:
            data = doc.to_dict()
            known_faces.append({
                "name": data["name"],
                "embedding": np.array(data["embedding"]) 
            })
    return known_faces

# ================= API 1: ĐĂNG KÝ (Hỗ trợ cả File và URL) =================
@app.route('/register', methods=['POST'])
def register_face():
    frame = None
    name = None

    # Cách 1: Gửi qua Link (Thunkable gửi JSON)
    if request.is_json:
        data = request.get_json()
        if 'url' in data:
            print("Đang tải ảnh đăng ký từ URL...")
            frame = url_to_image(data['url'])
        if 'name' in data:
            name = data['name']

    # Cách 2: Gửi qua File trực tiếp
    elif 'file' in request.files:
        file = request.files['file']
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if 'name' in request.form:
            name = request.form['name']

    # Kiểm tra dữ liệu đầu vào
    if frame is None:
        return jsonify({"error": "Không nhận được ảnh (hoặc URL lỗi)"}), 400
    if name is None:
        return jsonify({"error": "Thiếu tên người dùng"}), 400
    
    # Xử lý AI
    emb = get_embedding(frame)
    
    # Dọn rác ngay
    del frame
    gc.collect()

    if emb is None:
        return jsonify({"error": "Không tìm thấy khuôn mặt trong ảnh"}), 400
        
    try:
        db.collection(COLLECTION_FACES).document(name).set({
            "name": name,
            "embedding": emb.tolist(),
            "created_at": firestore.SERVER_TIMESTAMP
        })
        return jsonify({"status": "success", "message": f"Registered {name}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================= API 2: NHẬN DIỆN (Hỗ trợ cả File và URL) =================
@app.route('/detect', methods=['POST'])
def detect_face():
    frame = None

    # Cách 1: Gửi qua Link (Thunkable gửi JSON)
    if request.is_json and 'url' in request.json:
        print("Đang tải ảnh nhận diện từ URL...")
        frame = url_to_image(request.json['url'])
    
    # Cách 2: Gửi qua File
    elif 'file' in request.files:
        file = request.files['file']
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if frame is None:
        return jsonify({"error": "Không nhận được ảnh"}), 400
    
    # 1. Lấy vector của mặt người đang login
    target_emb = get_embedding(frame)
    
    # Dọn rác ảnh ngay
    del frame
    gc.collect()

    if target_emb is None:
        return jsonify({"status": "no_face_found"}), 200
        
    # 2. Tải danh sách người đã đăng ký về
    known_faces = load_known_faces()
    if not known_faces:
         return jsonify({"status": "unknown", "message": "Database empty"}), 200
         
    # 3. So sánh khoảng cách
    min_dist = 100.0
    identified_name = "Unknown"
    
    for face in known_faces:
        dist = np.sqrt(np.sum(np.square(target_emb - face["embedding"])))
        if dist < min_dist:
            min_dist = dist
            if dist < DISTANCE_THRESHOLD:
                identified_name = face["name"]
                
    # 4. Trả kết quả
    result = {
        "name": identified_name,
        "distance": float(min_dist),
        "status": "success" if identified_name != "Unknown" else "unknown"
    }
    
    if identified_name != "Unknown":
         db.collection(LOG_COLLECTION).add({
             "name": identified_name,
             "timestamp": firestore.SERVER_TIMESTAMP
         })
         
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)