# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import sys
import threading
import time
from typing import Optional, Tuple, Dict
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
    # 使用 tflite_runtime
    import tflite_runtime.interpreter as tflite
    print("[INFO] 成功匯入 tflite_runtime 函式庫")
except ImportError:
    # 如果找不到 tflite_runtime，退回使用 TensorFlow
    print("[WARN] 找不到 tflite_runtime，改用 TensorFlow。")
    import tensorflow.lite.python.interpreter as tflite
    
# ======== 模型路徑與設定 ========
SAD_EMOTION_MODEL_PATH = '/root/output/best_emotion_model_vela.tflite'
HAPPY_MODEL_PATH = '/root/output/best_happy_model_vela.tflite'
DEMOGRAPHIC_MODEL_PATH = '/root/output/best_mobilenet_finetuned_model_vela.tflite'
FACEDETECTOR_MODEL_PATH = '/root/face_detector.tflite'
FONT_PATH = "NotoSerifCJKtc-Regular.otf"

SAD_IMG_SIZE = (48, 48)
HAPPY_IMG_SIZE = (96, 96)
AGE_GENDER_IMG_SIZE = (128, 128)

SAD_EMOTION_MAP = {0: 'not_sad', 1: 'sad'}
HAPPY_EMOTION_MAP = {0: 'happy', 1: 'not_happy'}
GENDER_MAP = {0: 'male', 1: 'female'}

# 將年齡轉換為年齡區間標籤
def age_to_bin(age):
    age = int(age)
    if 18 <= age <= 30: return '18-30'
    elif 31 <= age <= 50: return '31-50'
    elif age >= 51: return '51-above'
    else: return '0-17'

# 降低推論頻率以穩定結果，每6幀進行一次分析
INFERENCE_INTERVAL = 6

# 共享變數與鎖
latest_frame: Optional[np.ndarray] = None
lock = threading.Lock()
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
_interpreters = threading.local()

# ======== 攝影機設定與擷取執行緒 ========
def open_camera(cam_index: int):
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

# ======== 載入 TFLite 模型 (每個執行緒獨立載入) ========
def _load_tflite_model_instance(model_path: str):
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"[ERROR] 模型 {os.path.basename(model_path)} 載入失敗: {e}")
        return None

def get_interpreters():
    if not hasattr(_interpreters, 'cache'):
        _interpreters.cache = {
            'sad_emotion': _load_tflite_model_instance(SAD_EMOTION_MODEL_PATH),
            'happy': _load_tflite_model_instance(HAPPY_MODEL_PATH),
            'age_gender': _load_tflite_model_instance(DEMOGRAPHIC_MODEL_PATH)
        }
        print(f"[INFO] 執行緒 {threading.current_thread().name} 成功載入所有模型。")
    return _interpreters.cache

sad_emotion_interpreter_dummy = _load_tflite_model_instance(SAD_EMOTION_MODEL_PATH)
happy_interpreter_dummy = _load_tflite_model_instance(HAPPY_MODEL_PATH)
age_gender_interpreter_dummy = _load_tflite_model_instance(DEMOGRAPHIC_MODEL_PATH)

if not sad_emotion_interpreter_dummy or not happy_interpreter_dummy or not age_gender_interpreter_dummy:
    sys.exit(1)

# 初始化 MediaPipe 人臉偵測器
face_options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=FACEDETECTOR_MODEL_PATH),
    min_detection_confidence=0.7
)
face_detector = FaceDetector.create_from_options(face_options)
print("[INFO] MediaPipe 人臉偵測器準備完成！")
            
def perform_inference(face_crop: np.ndarray):
    results = {}
    try:
        interpreters = get_interpreters()
        if not all(interpreters.values()):
            print("[ERROR] 推論執行緒無法取得解譯器。")
            return {}
            
        sad_emotion_interpreter = interpreters['sad_emotion']
        happy_interpreter = interpreters['happy']
        age_gender_interpreter = interpreters['age_gender']
        
        sad_input_details = sad_emotion_interpreter.get_input_details()
        sad_output_details = sad_emotion_interpreter.get_output_details()
        sad_input_data = np.expand_dims(cv2.resize(face_crop, SAD_IMG_SIZE) / 255.0, axis=0).astype(np.float32)
        sad_emotion_interpreter.set_tensor(sad_input_details[0]['index'], sad_input_data)
        sad_emotion_interpreter.invoke()
        sad_preds = sad_emotion_interpreter.get_tensor(sad_output_details[0]['index'])
        
        happy_input_details = happy_interpreter.get_input_details()
        happy_output_details = happy_interpreter.get_output_details()
        happy_input_data = np.expand_dims(cv2.resize(face_crop, HAPPY_IMG_SIZE) / 255.0, axis=0).astype(np.float32)
        happy_interpreter.set_tensor(happy_input_details[0]['index'], happy_input_data)
        happy_interpreter.invoke()
        happy_preds = happy_interpreter.get_tensor(happy_output_details[0]['index'])

        age_gender_input_details = age_gender_interpreter.get_input_details()
        age_gender_output_details = age_gender_interpreter.get_output_details()
        age_gender_input_data = np.expand_dims(cv2.resize(face_crop, AGE_GENDER_IMG_SIZE) / 255.0, axis=0).astype(np.float32)
        age_gender_interpreter.set_tensor(age_gender_input_details[0]['index'], age_gender_input_data)
        age_gender_interpreter.invoke()
        
        gender_preds = age_gender_interpreter.get_tensor(age_gender_output_details[0]['index'])
        age_preds = age_gender_interpreter.get_tensor(age_gender_output_details[1]['index'])

        gender_prob = gender_preds[0][0]
        current_gender_idx = 1 if gender_prob > 0.5 else 0
        results['gender_label'] = GENDER_MAP[current_gender_idx]
        results['gender_confidence'] = gender_prob if current_gender_idx == 1 else 1 - gender_prob

        sad_prob = sad_preds[0][0]
        current_sad_idx = 1 if sad_prob > 0.5 else 0
        results['sad_label'] = SAD_EMOTION_MAP[current_sad_idx]
        results['sad_confidence'] = sad_prob if current_sad_idx == 1 else 1 - sad_prob
        
        happy_prob = happy_preds[0][0]
        current_happy_idx = 0 if happy_prob < 0.5 else 1
        results['happy_label'] = HAPPY_EMOTION_MAP[current_happy_idx]
        results['happy_confidence'] = happy_prob if current_happy_idx == 1 else 1 - happy_prob

        if len(age_preds[0]) > 0:
            age_value = age_preds[0][0]
            results['age_value'] = age_value
            
    except Exception as e:
        print(f"[ERROR] 推論執行緒發生錯誤: {e}")
        return {}
    
    return results

# ======== 主程式初始化 ========
def main():
    cam_index = 0
    cap = open_camera(cam_index)
    if cap is None:
        print(f"[ERROR] 無法開啟攝影機索引 {cam_index}，請檢查裝置。")
        sys.exit(1)
    
    threading.Thread(target=camera_grabber, args=(cap,), daemon=True).start()
    
    print("[INFO] 等待攝影機畫面準備...")
    while latest_frame is None:
        time.sleep(0.1)
    
    # 新增：初始化 AdRecommender 類別
    ad_recommender = AdRecommender()
    ad_recommender.initialize_first_ad()
    
    cam_h, cam_w = latest_frame.shape[:2]
    sys_info_height = 60
    ad_h = cam_h + sys_info_height
    ad_w = int(ad_h * (600.0 / 900.0))
    
    cv2.setUseOptimized(False)
    window_name = 'Emotion Ad System'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    combined_w = cam_w + ad_w
    combined_h = cam_h + sys_info_height
    cv2.resizeWindow(window_name, combined_w, combined_h)
    print(f"[INFO] 介面視窗大小設定為 {combined_w}x{combined_h}")
    
    try:
        font_label = ImageFont.truetype(FONT_PATH, 24)
    except IOError:
        font_label = ImageFont.load_default()
    print("[INFO] 介面字型載入完成")
    
    print("[INFO] 系統啟動完成！按 'q' 鍵退出。")

    frame_count = 0
    next_face_id: int = 0
    cpu_usage = 0.0
    face_bboxes: Dict[int, Tuple] = {}
    face_track_data: Dict[int, Dict] = {}
    future_results: Dict[int, concurrent.futures.Future] = {} 

    # ======== 主迴圈 ========
    while True:
        with lock:
            frame = latest_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue
        H, W = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = face_detector.detect(mp_image)
        
        if frame_count % 30 == 0:
            cpu_usage = psutil.cpu_percent(interval=1)
        
        new_face_bboxes = {}
        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                x1, y1, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                new_box = (x1, y1, x1 + w, y1 + h)
                
                best_match_id = -1
                max_iou = 0
                for face_id, old_box in face_bboxes.items():
                    iou = (max(0, min(new_box[2], old_box[2]) - max(new_box[0], old_box[0])) * max(0, min(new_box[3], old_box[3]) - max(new_box[1], old_box[1]))) / float((new_box[2] - new_box[0]) * (new_box[3] - new_box[1]) + (old_box[2] - old_box[0]) * (old_box[3] - old_box[1]) - max(0, min(new_box[2], old_box[2]) - max(new_box[0], old_box[0])) * max(0, min(new_box[3], old_box[3]) - max(new_box[1], old_box[1])))
                    if iou > max_iou:
                        max_iou = iou
                        best_match_id = face_id
                
                if max_iou > 0.4:
                    face_id = best_match_id
                else:
                    face_id = next_face_id
                    next_face_id += 1
                
                new_face_bboxes[face_id] = new_box
                
                if frame_count % INFERENCE_INTERVAL == 0:
                    face_bgr = frame[y1:y1+h, x1:x1+w]
                    if face_bgr.size > 0:
                        face_rgb_crop = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                        future_results[face_id] = executor.submit(perform_inference, face_rgb_crop)
        
        face_bboxes = new_face_bboxes
        
        for face_id in list(future_results.keys()):
            future = future_results[face_id]
            if future.done():
                try:
                    new_results = future.result()
                    if new_results:
                        if face_id not in face_track_data:
                            face_track_data[face_id] = {
                                'gender_history': deque(maxlen=10),
                                'age_history': deque(maxlen=15),
                                'sad_history': deque(maxlen=10),
                                'happy_history': deque(maxlen=10),
                                'stable_gender': 'Analyzing...',
                                'stable_age_bin': 'Analyzing...',
                                'stable_sad_emotion': 'Analyzing...',
                                'stable_happy_emotion': 'Analyzing...',
                                'is_looking_at_screen': True
                            }

                        current_data = face_track_data[face_id]
                        current_data['gender_history'].append(new_results['gender_label'])
                        current_data['age_history'].append(new_results['age_value'])
                        current_data['sad_history'].append(new_results['sad_label'])
                        current_data['happy_history'].append(new_results['happy_label'])

                        gender_counts = Counter(current_data['gender_history'])
                        most_common_gender = gender_counts.most_common(1)
                        if most_common_gender and most_common_gender[0][1] >= 5:
                            current_data['stable_gender'] = most_common_gender[0][0]

                        sad_counts = Counter(current_data['sad_history'])
                        most_common_sad = sad_counts.most_common(1)
                        if most_common_sad and most_common_sad[0][1] >= 5:
                            current_data['stable_sad_emotion'] = most_common_sad[0][0]
                        
                        happy_counts = Counter(current_data['happy_history'])
                        most_common_happy = happy_counts.most_common(1)
                        if most_common_happy and most_common_happy[0][1] >= 5:
                            current_data['stable_happy_emotion'] = most_common_happy[0][0]

                        if current_data['age_history']:
                            median_age = np.median(list(current_data['age_history']))
                            current_data['stable_age_bin'] = age_to_bin(median_age)
                except Exception as e:
                    print(f"[ERROR] 處理執行緒結果時發生錯誤: {e}")
                finally:
                    del future_results[face_id]
        
        # 取得主要人臉資訊並傳給 AdRecommender
        main_face_info = {}
        main_face_id = -1
        max_area = 0
        for face_id, bbox in face_bboxes.items():
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > max_area:
                max_area = area
                main_face_id = face_id
        if main_face_id in face_track_data:
            main_face_info = face_track_data[main_face_id]

        # 新增：從 AdRecommender 取得廣告幀
        ad_frame = ad_recommender.get_ad_frame(main_face_info, ad_w, ad_h)
        
        # 繪製所有邊界框
        processed_frame = frame.copy()
        for face_id, bbox in face_bboxes.items():
            x1, y1, x2, y2 = bbox
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            face_info = face_track_data.get(face_id, {
                'stable_gender': '...', 'stable_age_bin': '...',
                'stable_sad_emotion': '...', 'stable_happy_emotion': '...'
            })
            label_lines = [
                f"ID:{face_id}",
                f"性別: {face_info['stable_gender']}", 
                f"年齡: {face_info['stable_age_bin']}",
                f"情緒: {face_info['stable_happy_emotion']} / {face_info['stable_sad_emotion']}"
            ]
            pil_img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            text_x = x1
            text_y = y1 - 40 if y1 >= 40 else y2 + 5
            try:
                font_label_small = ImageFont.truetype(FONT_PATH, 18)
            except IOError:
                font_label_small = ImageFont.load_default()
            for line in label_lines:
                draw.text((text_x, text_y), line, fill=(20, 255, 20), font=font_label_small)
                text_y += 20
            processed_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
        # 繪製系統資訊
        sys_info_lines = [
            f"CPU 使用率: {psutil.cpu_percent(interval=1):.1f}%",
            f"目前廣告類別: {ad_recommender.current_ad_category}"
        ]
        sys_info_bg = np.zeros((sys_info_height, W, 3), np.uint8)
        pil_sys_info = Image.fromarray(cv2.cvtColor(sys_info_bg, cv2.COLOR_BGR2RGB))
        draw_sys = ImageDraw.Draw(pil_sys_info)
        try:
            font_sys_info = ImageFont.truetype(FONT_PATH, 20)
        except IOError:
            font_sys_info = ImageFont.load_default()
        text_y = 10
        for line in sys_info_lines:
            text_x = 10
            draw_sys.text((text_x, text_y), line, fill=(255, 255, 255), font=font_sys_info)
            text_y += 25
        sys_info_frame = cv2.cvtColor(np.array(pil_sys_info), cv2.COLOR_RGB2BGR)
        processed_frame = np.vstack((processed_frame, sys_info_frame))
    
        # 組合畫面
        if ad_frame is not None:
            combined_frame = np.hstack((processed_frame, ad_frame))
        else:
            black_ad_frame = np.zeros((H + sys_info_height, ad_w, 3), dtype=np.uint8)
            combined_frame = np.hstack((processed_frame, black_ad_frame))
        cv2.line(combined_frame, (W, 0), (W, combined_frame.shape[0]), (255, 255, 255), 2)
        cv2.imshow(window_name, combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()