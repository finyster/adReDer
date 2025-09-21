# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import sys
import threading
import time
from typing import Optional, Tuple, Dict, List
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions
from collections import deque, Counter
import concurrent.futures
import psutil

# 新增：從 ad_recommender.py 匯入 AdRecommender 類別
from ad_recommender import AdRecommender

# ======== 偵測/推論相依性 ========
try:
    import tflite_runtime.interpreter as tflite
    print("[INFO] 成功匯入 tflite_runtime 函式庫")
except ImportError:
    print("[WARN] 找不到 tflite_runtime，改用 TensorFlow。")
    import tensorflow.lite.python.interpreter as tflite

# ======== 全域常數與設定 (Global Constants & Configuration) ========
# --- 模型路徑 ---
SAD_EMOTION_MODEL_PATH = '/root/output/best_emotion_model_vela.tflite'
HAPPY_MODEL_PATH = '/root/output/best_happy_model_vela.tflite'
DEMOGRAPHIC_MODEL_PATH = '/root/output/best_mobilenet_finetuned_model_vela.tflite'
FACEDETECTOR_MODEL_PATH = '/root/face_detector.tflite'
FONT_PATH = "NotoSerifCJKtc-Regular.otf"

# --- 模型設定 ---
SAD_IMG_SIZE = (48, 48)
HAPPY_IMG_SIZE = (96, 96)
AGE_GENDER_IMG_SIZE = (128, 128)
SAD_EMOTION_MAP = {0: 'not_sad', 1: 'sad'}
HAPPY_EMOTION_MAP = {0: 'happy', 1: 'not_happy'}
# <<< 注意：根據你的模型，0 是 Male, 1 是 Female
GENDER_MAP = {0: 'Male', 1: 'Female'} 

# --- 效能與追蹤設定 ---
INFERENCE_INTERVAL = 6      
IOU_THRESHOLD = 0.4         
MAX_AGE = 30                
HISTORY_LEN = 15            
STABILITY_THRESHOLD = 5     

# 共享變數與執行緒池
latest_frame: Optional[np.ndarray] = None
lock = threading.Lock()
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
_interpreters = threading.local()

# ======== 核心類別：FaceTracker ========
class Face:
    """用於儲存單一人臉的所有追蹤資訊"""
    def __init__(self, face_id: int, bbox: Tuple[int, int, int, int]):
        self.id = face_id
        self.bbox = bbox
        self.frames_since_seen = 0
        self.future_result: Optional[concurrent.futures.Future] = None
        
        # 歷史數據
        self.age_history = deque(maxlen=HISTORY_LEN)
        self.sad_history = deque(maxlen=HISTORY_LEN)
        self.happy_history = deque(maxlen=HISTORY_LEN)
        
        # 穩定結果
        self.stable_age_bin = "Analyzing..."
        self.stable_gender = "Analyzing..."
        self.stable_sad_emotion = "Analyzing..."
        self.stable_happy_emotion = "Analyzing..."
        self.exact_age = 0.0
        
        # <<< MODIFIED: 新增 gender_confidence 來儲存最高信心度
        self.gender_confidence = 0.0

    def update_bbox(self, bbox: Tuple[int, int, int, int]):
        self.bbox = bbox
        self.frames_since_seen = 0

    def _update_stable_emotion(self, history: deque, attribute_name: str):
        if len(history) > STABILITY_THRESHOLD:
            counts = Counter(history)
            most_common = counts.most_common(1)
            if most_common and most_common[0][1] >= STABILITY_THRESHOLD:
                setattr(self, attribute_name, most_common[0][0])

    def update_results(self, results: Dict):
        # 年齡判斷：取歷史最大值
        if 'age_value' in results:
            self.age_history.append(results['age_value'])
            self.exact_age = max(self.age_history)
            self.stable_age_bin = age_to_bin(self.exact_age)

        # <<< MODIFIED: 性別判斷：採用最高信心度策略
        if 'gender_prob' in results:
            gender_prob = results['gender_prob']
            current_gender_idx = 1 if gender_prob > 0.5 else 0
            # 計算當前預測的信心度
            confidence = gender_prob if current_gender_idx == 1 else 1 - gender_prob
            
            # 如果當前信心度更高，則更新結果
            if confidence > self.gender_confidence:
                self.gender_confidence = confidence
                self.stable_gender = GENDER_MAP[current_gender_idx]

        # 情緒判斷：維持多數決，因為情緒變化較快
        if 'sad_label' in results:
            self.sad_history.append(results['sad_label'])
            self._update_stable_emotion(self.sad_history, 'stable_sad_emotion')
        if 'happy_label' in results:
            self.happy_history.append(results['happy_label'])
            self._update_stable_emotion(self.happy_history, 'stable_happy_emotion')

class FaceTracker:
    # ... (此類別不變)
    def __init__(self):
        self.faces: Dict[int, Face] = {}
        self.next_face_id = 0

    def _get_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def update(self, new_detections: List[Tuple[int, int, int, int]]):
        for face in self.faces.values():
            face.frames_since_seen += 1

        matched_indices = set()
        for face_id, face in self.faces.items():
            best_match_iou = 0
            best_match_idx = -1
            for i, new_box in enumerate(new_detections):
                if i in matched_indices: continue
                iou = self._get_iou(face.bbox, new_box)
                if iou > best_match_iou:
                    best_match_iou = iou
                    best_match_idx = i
            
            if best_match_iou > IOU_THRESHOLD:
                face.update_bbox(new_detections[best_match_idx])
                matched_indices.add(best_match_idx)

        for i, new_box in enumerate(new_detections):
            if i not in matched_indices:
                new_face = Face(self.next_face_id, new_box)
                self.faces[self.next_face_id] = new_face
                self.next_face_id += 1
        
        lost_ids = [face_id for face_id, face in self.faces.items() if face.frames_since_seen > MAX_AGE]
        for face_id in lost_ids:
            del self.faces[face_id]

# ======== 輔助函式 ========
def age_to_bin(age):
    age = int(age)
    if 18 <= age <= 30: return '18-30'
    elif 31 <= age <= 50: return '31-50'
    elif age >= 51: return '51-above'
    else: return '0-17'

def open_camera(cam_index: int):
    # ... (此函式不變)
    gst = (
        f"v4l2src device=/dev/video{cam_index} io-mode=2 ! "
        "image/jpeg,framerate=30/1 ! jpegdec ! "
        "videoscale ! video/x-raw,width=1280,height=720 ! "
        "queue leaky=2 max-size-buffers=2 ! "
        "videoconvert ! appsink drop=true max-buffers=1 sync=false"
    )
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print(f"[INFO] 成功使用 GStreamer 開啟攝影機 {cam_index} (1280x720)")
        return cap
    print(f"[WARN] GStreamer 開啟失敗，退回標準 V4L2。")
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap
    return None

def camera_grabber(cap: cv2.VideoCapture):
    # ... (此函式不變)
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 無法從攝影機讀取影像。")
            time.sleep(0.01)
            continue
        with lock:
            latest_frame = frame.copy()
        time.sleep(0.005)

# ======== 推論與繪圖 ========
def _load_tflite_model_instance(model_path: str):
    # ... (此函式不變)
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"[ERROR] 模型 {os.path.basename(model_path)} 載入失敗: {e}")
        return None

def get_interpreters():
    # ... (此函式不變)
    if not hasattr(_interpreters, 'cache'):
        _interpreters.cache = {
            'sad_emotion': _load_tflite_model_instance(SAD_EMOTION_MODEL_PATH),
            'happy': _load_tflite_model_instance(HAPPY_MODEL_PATH),
            'age_gender': _load_tflite_model_instance(DEMOGRAPHIC_MODEL_PATH)
        }
        print(f"[INFO] 執行緒 {threading.current_thread().name} 成功載入所有模型。")
    return _interpreters.cache

def perform_inference(face_crop: np.ndarray):
    results = {}
    try:
        interpreters = get_interpreters()
        sad_emotion_interpreter = interpreters['sad_emotion']
        happy_interpreter = interpreters['happy']
        age_gender_interpreter = interpreters['age_gender']
        
        # ... (情緒推論部分不變) ...
        # Sad Emotion
        sad_input_details = sad_emotion_interpreter.get_input_details()
        sad_output_details = sad_emotion_interpreter.get_output_details()
        sad_input_data = np.expand_dims(cv2.resize(face_crop, SAD_IMG_SIZE) / 255.0, axis=0).astype(np.float32)
        sad_emotion_interpreter.set_tensor(sad_input_details[0]['index'], sad_input_data)
        sad_emotion_interpreter.invoke()
        sad_preds = sad_emotion_interpreter.get_tensor(sad_output_details[0]['index'])
        results['sad_label'] = SAD_EMOTION_MAP[1 if sad_preds[0][0] > 0.5 else 0]

        # Happy Emotion
        happy_input_details = happy_interpreter.get_input_details()
        happy_output_details = happy_interpreter.get_output_details()
        happy_input_data = np.expand_dims(cv2.resize(face_crop, HAPPY_IMG_SIZE) / 255.0, axis=0).astype(np.float32)
        happy_interpreter.set_tensor(happy_input_details[0]['index'], happy_input_data)
        happy_interpreter.invoke()
        happy_preds = happy_interpreter.get_tensor(happy_output_details[0]['index'])
        results['happy_label'] = HAPPY_EMOTION_MAP[1 if happy_preds[0][0] > 0.5 else 0]
        
        # Age & Gender
        age_gender_input_details = age_gender_interpreter.get_input_details()
        # <<< 注意：你的模型輸出順序是 age, gender
        age_gender_output_details = age_gender_interpreter.get_output_details()
        age_gender_input_data = np.expand_dims(cv2.resize(face_crop, AGE_GENDER_IMG_SIZE) / 255.0, axis=0).astype(np.float32)
        age_gender_interpreter.set_tensor(age_gender_input_details[0]['index'], age_gender_input_data)
        age_gender_interpreter.invoke()
        
        # <<< MODIFIED: 根據你的模型輸出順序調整
        age_preds = age_gender_interpreter.get_tensor(age_gender_output_details[0]['index'])
        gender_preds = age_gender_interpreter.get_tensor(age_gender_output_details[1]['index'])

        # <<< MODIFIED: 同時回傳 age, gender_prob
        if len(age_preds[0]) > 0:
            results['age_value'] = age_preds[0][0]
        results['gender_prob'] = gender_preds[0][0]
            
    except Exception as e:
        print(f"[ERROR] 推論執行緒發生錯誤: {e}")
        return {}
    
    return results

def draw_results(frame: np.ndarray, tracker: FaceTracker, sys_info: Dict):
    """統一繪製所有結果到畫面上"""
    for face in tracker.faces.values():
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # <<< MODIFIED: 顯示性別和其最高信心度
        gender_display = f"{face.stable_gender} ({face.gender_confidence*100:.0f}%)" if face.stable_gender != "Analyzing..." else "Analyzing..."
        age_display = f"{face.exact_age:.0f} ({face.stable_age_bin})" if face.stable_age_bin != "Analyzing..." else "Analyzing..."

        label_y = y1 - 10 if y1 > 30 else y2 + 20
        cv2.putText(frame, f"ID:{face.id}", (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Gender: {gender_display}", (x1, label_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Age: {age_display}", (x1, label_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Emotion: {face.stable_happy_emotion}/{face.stable_sad_emotion}", (x1, label_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 繪製系統資訊
    sys_info_bg = np.zeros((60, frame.shape[1], 3), np.uint8)
    cv2.putText(sys_info_bg, f"CPU Usage: {sys_info.get('cpu_usage', 0):.1f}%", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(sys_info_bg, f"Current Ad: {sys_info.get('ad_category', 'N/A')}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return np.vstack((frame, sys_info_bg))

# ======== 主程式 (Main Application) ========
def main():
    # ... (此函式不變)
    cam_index = 0
    cap = open_camera(cam_index)
    if cap is None:
        print(f"[ERROR] 無法開啟攝影機索引 {cam_index}，請檢查裝置。")
        sys.exit(1)
    
    threading.Thread(target=camera_grabber, args=(cap,), daemon=True).start()
    
    print("[INFO] 等待攝影機畫面準備...")
    while latest_frame is None:
        time.sleep(0.1)
    
    ad_recommender = AdRecommender()
    if not ad_recommender.ad_data:
        print("[WARN] 沒有載入推薦資料，將使用隨機廣告類別。")
    ad_recommender.initialize_first_ad()
    tracker = FaceTracker()

    face_options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=FACEDETECTOR_MODEL_PATH),
        min_detection_confidence=0.7
    )
    face_detector = FaceDetector.create_from_options(face_options)
    print("[INFO] MediaPipe 人臉偵測器準備完成！")
    
    cam_h, cam_w = latest_frame.shape[:2]
    sys_info_height = 60
    ad_h = cam_h + sys_info_height
    ad_w = int(ad_h * (600.0 / 900.0))
    window_name = 'Emotion Ad System'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, cam_w + ad_w, ad_h)
    
    print("[INFO] 系統啟動完成！按 'q' 鍵退出。")

    frame_count = 0
    psutil.cpu_percent(interval=None) 

    while True:
        with lock:
            if latest_frame is None: continue
            frame = latest_frame.copy()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = face_detector.detect(mp_image)
        
        current_detections = []
        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                x1, y1, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                current_detections.append((x1, y1, x1 + w, y1 + h))

        tracker.update(current_detections)

        for face in tracker.faces.values():
            if frame_count % INFERENCE_INTERVAL == 0 and face.future_result is None:
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                face_crop_bgr = frame[y1:y2, x1:x2]
                if face_crop_bgr.size > 0:
                    face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                    face.future_result = executor.submit(perform_inference, face_crop_rgb)
            
            if face.future_result and face.future_result.done():
                try:
                    results = face.future_result.result()
                    if results:
                        face.update_results(results)
                except Exception as e:
                    print(f"[ERROR] 處理 Face ID {face.id} 的結果時出錯: {e}")
                finally:
                    face.future_result = None
        
        main_face_info = {}
        if tracker.faces:
            main_face = max(tracker.faces.values(), key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            main_face_info = {
                'stable_gender': main_face.stable_gender,
                'stable_age_bin': main_face.stable_age_bin,
                'stable_sad_emotion': main_face.stable_sad_emotion,
                'stable_happy_emotion': main_face.stable_happy_emotion,
                'is_looking': True
            }
        ad_frame = ad_recommender.get_ad_frame(main_face_info, ad_w, ad_h)

        sys_info = {
            'cpu_usage': psutil.cpu_percent(interval=None),
            'ad_category': ad_recommender.current_ad_category
        }
        processed_frame = draw_results(frame, tracker, sys_info)

        if ad_frame is not None:
            combined_frame = np.hstack((processed_frame, ad_frame))
        else:
            black_ad_frame = np.zeros((ad_h, ad_w, 3), dtype=np.uint8)
            combined_frame = np.hstack((processed_frame, black_ad_frame))
        
        cv2.line(combined_frame, (cam_w, 0), (cam_w, combined_frame.shape[0]), (255, 255, 255), 2)
        cv2.imshow(window_name, combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown()
    
if __name__ == '__main__':
    # 確保主執行緒載入模型以供檢查
    get_interpreters()
    main()