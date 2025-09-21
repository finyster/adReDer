import numpy as np

class AdRecommender:
    def __init__(self, preferences_data, weights_data):
        """
        初始化推薦器
        :param preferences_data: 包含所有特徵偏好度分數的字典
        :param weights_data: 包含各特徵權重的字典
        """
        self.preferences = preferences_data
        self.weights = weights_data
        print("AdRecommender initialized successfully.")

    def get_recommendation(self, people_list):
        """
        根據在場所有人的特徵，計算出最佳廣告
        :param people_list: 一個包含多個 'person' 字典的列表
        :return: 一個包含最佳廣告名稱和所有廣告分數的字典
        """
        
        # 取得所有可能的廣告類別
        ad_categories = set()
        for gender_data in self.preferences['age_gender'].values():
            for age_data in gender_data.values():
                ad_categories.update(age_data.keys())
        
        total_scores = {category: 0.0 for category in ad_categories}

        # 1. 遍歷畫面中的每一個人
        for person in people_list:
            features = person.get('features', {})
            
            # 2. 計算 age_gender 分數
            gender = features.get('gender')
            age_group = features.get('age_group')
            if gender and age_group:
                score_table = self.preferences['age_gender'].get(gender, {}).get(age_group, {})
                weight = self.weights.get('age_gender', 1.0)
                for category, score in score_table.items():
                    total_scores[category] += score * weight

            # 3. 計算 interest (is_looking) 分數
            is_looking = features.get('is_looking')
            if is_looking is not None:
                # 將布林值轉為字串 "true" 或 "false" 來匹配 JSON key
                score_table = self.preferences['interest']['is_looking'].get(str(is_looking).lower(), {})
                weight = self.weights.get('interest', 1.0)
                # "*" 代表對所有類別都生效
                if "*" in score_table:
                    universal_score = score_table["*"]
                    for category in total_scores:
                        total_scores[category] += universal_score * weight

            # 注意：emotion_content 的邏輯我們先不加，因為它推薦的是「內容」而非「廣告」，
            # 我們可以在之後的階段再把它整合進來。

        if not total_scores:
            return {"best_ad": None, "scores": {}}

        # 4. 找到分數最高的廣告
        best_ad = max(total_scores, key=total_scores.get)
        
        return {
            "best_ad": best_ad,
            "scores": total_scores
        }