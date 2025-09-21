# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import json
import random
import time
from collections import Counter
from typing import Dict, List, Optional
from PIL import Image, ImageDraw, ImageFont

# ======== 邏輯類別 → 實體資料夾對應 ========
LOGICAL_TO_PHYSICAL = {
    "遊戲相關廣告": "game",
    "服裝相關廣告": "dress",
    "化妝品、保養品相關廣告": "makeup",
    "汽車相關廣告": "auto",
    "電腦相關廣告": "computer",
    "球類運動相關廣告": "ball",
    "保健品相關": "health",
    "旅遊相關廣告": "travel",
    "演唱會相關廣告": "asong",
    "展覽、快展活動相關廣告": "behavior",
    # 情緒類
    "旅行、冒險、休閒活動": "travel",
    "美食、飲料推薦": "health",
    "有趣幽默的短片": "game",
    "輕鬆搞笑的短影音": "game",
    "美食甜點": "health",
    "可愛動物圖片或影片": "asong",
    # fallback
    "中性": "neutral",
}

# ======== 年齡區間正規化 ========
def normalize_age_bin(bin_from_model: str) -> str:
    if not bin_from_model:
        return "21-30"
    b = str(bin_from_model).strip()
    if b in {"11-20", "21-30", "31-40", "41-50", "51-60", "61-above"}:
        return b

    import re
    m = re.search(r"(\d+)", b)
    if m:
        age = int(m.group(1))
        if 11 <= age <= 20: return "11-20"
        if 21 <= age <= 30: return "21-30"
        if 31 <= age <= 40: return "31-40"
        if 41 <= age <= 50: return "41-50"
        if 51 <= age <= 60: return "51-60"
        if age >= 61: return "61-above"

    if "51" in b and ("+" in b or "above" in b):
        return "51-60"
    return "21-30"

# ======== 廣告媒體資料庫 ========
def create_ad_media_db(base_dir: str = 'ads') -> Dict[str, Dict[str, List[str]]]:
    ad_media_db = {'videos': {}, 'images': {}}
    IGNORE_DIRS = {'.git', '.svn', '__pycache__', 'FSCK0000.000'}

    if not os.path.exists(base_dir) or not os.listdir(base_dir):
        print(f"[WARN] 找不到 '{base_dir}' 或為空，建立預設 neutral 廣告。")
        os.makedirs(os.path.join(base_dir, 'neutral'), exist_ok=True)
        try:
            font_path = "NotoSerifCJKtc-Regular.otf"
            try:
                font = ImageFont.truetype(font_path, 60)
            except IOError:
                font = ImageFont.load_default()
            img = Image.new('RGB', (600, 900), (200, 200, 200))
            d = ImageDraw.Draw(img)
            d.multiline_text((50, 250), "Default Ad\nPut media in ads/", fill=(0, 0, 0), font=font, spacing=10)
            img.save(os.path.join(base_dir, 'neutral', 'neutral.jpg'))
        except Exception as e:
            print(f"[ERROR] 建立預設廣告失敗: {e}")

    video_exts = ('.mp4', '.mov', '.avi', '.mkv')
    image_exts = ('.jpg', '.jpeg', '.png')

    for category in sorted([d for d in os.listdir(base_dir)
                            if os.path.isdir(os.path.join(base_dir, d)) and d not in IGNORE_DIRS]):
        category_path = os.path.join(base_dir, category)
        vids, imgs = [], []
        for root, _, files in os.walk(category_path):
            for f in files:
                lower = f.lower()
                full = os.path.join(root, f)
                if lower.endswith(video_exts):
                    vids.append(full)
                elif lower.endswith(image_exts):
                    imgs.append(full)
        if vids:
            ad_media_db['videos'][category] = sorted(vids)
            print(f"[INFO] {category}: 影片 {len(vids)} 支")
        if imgs:
            ad_media_db['images'][category] = sorted(imgs)
            print(f"[INFO] {category}: 圖片 {len(imgs)} 張")

    return ad_media_db

# ======== 推薦分數計算 ========
def calculate_ad_scores(ad_data: dict, face_stats: dict,
                        available_categories: List[str]) -> Optional[str]:
    if not ad_data or not face_stats:
        return random.choice(available_categories) if available_categories else None

    logical_scores: Dict[str, float] = {}

    # 年齡 / 性別
    gender = (face_stats.get('stable_gender') or '').lower()
    age_bin = normalize_age_bin(face_stats.get('stable_age_bin') or '')
    age_gender_map = ad_data.get('age_gender', {})
    if gender in age_gender_map and age_bin in age_gender_map[gender]:
        for logical_cat, score in age_gender_map[gender][age_bin].items():
            logical_scores[logical_cat] = logical_scores.get(logical_cat, 0.0) + float(score)

    # 情緒
    stable_emotion = (face_stats.get('stable_emotion') or '').lower()
    emo_map = ad_data.get('emotion_content', {}).get(stable_emotion, {})
    for logical_cat, score in emo_map.items():
        logical_scores[logical_cat] = logical_scores.get(logical_cat, 0.0) + float(score)

    # 是否注視
    looking = bool(face_stats.get('is_looking'))
    look_map = ad_data.get('interest', {}).get('is_looking', {})
    if looking:
        bonus = float(look_map.get('true', {}).get('*', 0))
        for lc in list(logical_scores.keys()) or ['中性']:
            logical_scores[lc] = logical_scores.get(lc, 0.0) + bonus
    else:
        malus = float(look_map.get('false', {}).get('*', 0))
        for lc in list(logical_scores.keys()) or ['中性']:
            logical_scores[lc] = logical_scores.get(lc, 0.0) + malus

    # 映射到實體類別
    physical_scores: Dict[str, float] = {}
    for logical_cat, score in logical_scores.items():
        phys = LOGICAL_TO_PHYSICAL.get(logical_cat)
        if not phys:
            for c in available_categories:
                if logical_cat in c or c in logical_cat:
                    phys = c
                    break
        if phys and phys in available_categories:
            physical_scores[phys] = physical_scores.get(phys, 0.0) + score

    if not physical_scores:
        return 'neutral' if 'neutral' in available_categories else (
            available_categories[0] if available_categories else None
        )

    return max(physical_scores.items(), key=lambda kv: kv[1])[0]

# ======== 觀眾數據收集器 ========
class AdStatsCollector:
    def __init__(self):
        self.gender_counts = Counter()
        self.age_counts = Counter()
        self.sad_counts = Counter()
        self.happy_counts = Counter()
        self.frame_count = 0
        self.is_looking_count = 0

    def collect_data(self, face_info: dict):
        self.frame_count += 1
        if face_info.get('stable_gender') not in {None, 'Analyzing...'}:
            self.gender_counts[face_info['stable_gender']] += 1
        if face_info.get('stable_age_bin') not in {None, 'Analyzing...'}:
            self.age_counts[face_info['stable_age_bin']] += 1
        if face_info.get('stable_sad_emotion') not in {None, 'Analyzing...'}:
            self.sad_counts[face_info['stable_sad_emotion']] += 1
        if face_info.get('stable_happy_emotion') not in {None, 'Analyzing...'}:
            self.happy_counts[face_info['stable_happy_emotion']] += 1
        if face_info.get('is_looking_at_screen', True):
            self.is_looking_count += 1

    def get_summary(self):
        if self.frame_count == 0:
            return {}
        def _most_common(counter): return counter.most_common(1)[0][0] if counter else None
        summary = {
            'stable_gender': _most_common(self.gender_counts),
            'stable_age_bin': _most_common(self.age_counts),
            'stable_sad_emotion': _most_common(self.sad_counts),
            'stable_happy_emotion': _most_common(self.happy_counts),
            'is_looking': (self.is_looking_count / self.frame_count) > 0.5
        }
        if summary['stable_happy_emotion'] == 'happy' and summary['stable_sad_emotion'] == 'not_sad':
            summary['stable_emotion'] = 'happy'
        elif summary['stable_sad_emotion'] == 'sad' and summary['stable_happy_emotion'] == 'not_happy':
            summary['stable_emotion'] = 'sad'
        else:
            summary['stable_emotion'] = 'neutral'
        return summary

# ======== 廣告推薦管理類別 ========
class AdRecommender:
    def __init__(self, ad_media_path: str = 'ads', ad_data_path: str = 'ad_recommendation.json'):
        self.ad_media_db = create_ad_media_db(ad_media_path)
        try:
            with open(ad_data_path, 'r', encoding='utf-8') as f:
                self.ad_data = json.load(f)
            print("[INFO] 成功載入推薦資料。")
        except Exception as e:
            print(f"[ERROR] 載入推薦資料失敗: {e}")
            self.ad_data = None

        self.current_ad_cap: Optional[cv2.VideoCapture] = None
        self.current_ad_image: Optional[np.ndarray] = None
        self.ad_is_video = True
        self.ad_start_time = time.time()
        self.ad_duration = 0
        self.ad_stats_collector = AdStatsCollector()
        self.current_ad_category = 'neutral'
        self._cached_image_by_size = {}

    def _available_categories(self) -> List[str]:
        cats = set(self.ad_media_db['videos'].keys()) | set(self.ad_media_db['images'].keys())
        return [c for c in cats if (c in self.ad_media_db['videos'] and self.ad_media_db['videos'][c]) or
                                   (c in self.ad_media_db['images'] and self.ad_media_db['images'][c])]

    def start_ad(self, category: str):
        if self.current_ad_cap:
            self.current_ad_cap.release()
            self.current_ad_cap = None
        self.ad_stats_collector = AdStatsCollector()

        target_path, is_video_target = None, False
        if category in self.ad_media_db['videos'] and self.ad_media_db['videos'][category]:
            target_path = random.choice(self.ad_media_db['videos'][category])
            is_video_target = True
        elif category in self.ad_media_db['images'] and self.ad_media_db['images'][category]:
            target_path = random.choice(self.ad_media_db['images'][category])
            is_video_target = False

        if not target_path:
            print(f"[WARN] 類別 {category} 沒檔案，用 fallback。")
            target_path = self._get_fallback_category_path()
            if target_path:
                is_video_target = target_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))

        if not target_path:
            self.current_ad_image = np.zeros((900, 600, 3), dtype=np.uint8)
            self.ad_is_video = False
            self.ad_duration = 5
            self.current_ad_category = 'none'
            return

        self.current_ad_category = category
        self.ad_start_time = time.time()
        self.ad_is_video = is_video_target

        if self.ad_is_video:
            self.current_ad_cap = cv2.VideoCapture(target_path)
            if self.current_ad_cap.isOpened():
                fps = float(self.current_ad_cap.get(cv2.CAP_PROP_FPS) or 0.0)
                total = float(self.current_ad_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                self.ad_duration = (total / fps) if (fps > 1e-3 and total > 0) else 0.0
            else:
                print(f"[ERROR] 開啟影片失敗: {target_path}")
                self.ad_is_video = False
                self.start_ad('neutral')
        else:
            self.current_ad_image = cv2.imread(target_path)
            self.ad_duration = 10
            self._cached_image_by_size = {}

    def _get_fallback_category_path(self):
        for cat, files in self.ad_media_db['videos'].items():
            if files: return random.choice(files)
        for cat, files in self.ad_media_db['images'].items():
            if files: return random.choice(files)
        return None

    def get_ad_frame(self, face_stats: dict, ad_w: int, ad_h: int) -> np.ndarray:
        self.ad_stats_collector.collect_data(face_stats)
        ad_frame = None
        if self.ad_is_video and self.current_ad_cap:
            ret, ad_frame = self.current_ad_cap.read()
            if not ret:
                ad_frame = None
        else:
            key = (ad_w, ad_h)
            if key not in self._cached_image_by_size and self.current_ad_image is not None:
                self._cached_image_by_size[key] = cv2.resize(self.current_ad_image, (ad_w, ad_h))
            ad_frame = self._cached_image_by_size.get(key, np.zeros((ad_h, ad_w, 3), np.uint8))

        elapsed = time.time() - self.ad_start_time
        is_finished = False
        if self.ad_is_video:
            if ad_frame is None:  # EOF
                is_finished = True
        else:
            if self.ad_duration > 0 and elapsed > self.ad_duration:
                is_finished = True

        if is_finished:
            self._switch_ad()
            if self.ad_is_video and self.current_ad_cap:
                _, ad_frame = self.current_ad_cap.read()
            else:
                key = (ad_w, ad_h)
                if key not in self._cached_image_by_size and self.current_ad_image is not None:
                    self._cached_image_by_size[key] = cv2.resize(self.current_ad_image, (ad_w, ad_h))
                ad_frame = self._cached_image_by_size.get(key, np.zeros((ad_h, ad_w, 3), np.uint8))

        return cv2.resize(ad_frame, (ad_w, ad_h)) if ad_frame is not None else np.zeros((ad_h, ad_w, 3), np.uint8)

    def _switch_ad(self):
        print("[INFO] 廣告播放完，切換。")
        face_stats_summary = self.ad_stats_collector.get_summary()
        next_cat = calculate_ad_scores(self.ad_data, face_stats_summary, self._available_categories())
        if not next_cat:
            next_cat = 'neutral' if 'neutral' in self._available_categories() else None
        if next_cat:
            self.start_ad(next_cat)

    def initialize_first_ad(self):
        self.start_ad('neutral')
