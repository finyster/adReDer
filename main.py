import json
from core.recommender import AdRecommender

def load_config(path):
    """從 JSON 檔案載入設定"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    print("--- Starting Smart Ad System (Simulation Mode) ---")

    # 1. 載入設定檔
    print("Loading configuration files...")
    preferences = load_config('config/ad_preferences.json')
    weights = load_config('config/weights.json')

    # 2. 初始化推薦器
    recommender = AdRecommender(preferences_data=preferences, weights_data=weights)

    # 3. 建立假資料 (模擬攝影機偵測到的結果)
    print("\nSimulating detected people...")
    mock_people = [
        {'person_id': 1, 'features': {'age_group': '21-30', 'gender': 'female', 'is_looking': True}},
        {'person_id': 2, 'features': {'age_group': '31-40', 'gender': 'male', 'is_looking': True}},
        {'person_id': 3, 'features': {'age_group': '11-20', 'gender': 'male', 'is_looking': False}}
    ]
    print(f"Found {len(mock_people)} people.")

    # 4. 取得推薦結果
    print("\nCalculating best ad...")
    result = recommender.get_recommendation(mock_people)

    # 5. 印出結果
    print("\n--- Ad Recommendation Result ---")
    print(f"Best Ad to Display: {result['best_ad']}")
    print("\nDetailed Scores:")
    # 將分數由高到低排序後印出
    sorted_scores = sorted(result['scores'].items(), key=lambda item: item[1], reverse=True)
    for category, score in sorted_scores:
        print(f"- {category}: {score:.2f}")

if __name__ == "__main__":
    main()