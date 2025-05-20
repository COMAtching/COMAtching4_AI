import pandas as pd
import numpy as np
import argparse
import json
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

# 현재 디렉토리 상위 경로를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions import build_weighted_text_for_row

class CosineSimilarityRecommender:
    """
    MBTI + 연락빈도 + 대분류 취미 + 나이옵션 기반 코사인 유사도 추천 모델
    """
    def __init__(self):
        pass

    def recommend(self, user_profile, data, mbti_weight, contact_weight, hobby_weight, age_weight, top_k=5):
        """
        user_profile: {'mbtiOption': ..., 'contactfrequencyOption': ..., 'bigHobbyOption': ..., 'ageOption': ...}
        data: {
          'df': 필터링된 DataFrame,
          'tfidf_matrix': ...,
          'vectorizer': ...
        }
        """
        if data is None:
            return []

        if isinstance(data['df'], dict):
            data['df'] = pd.DataFrame(data['df'])

        if data['df'].empty:
            return []

        # 1) 유저 weighted text 생성
        pseudo_row = {
            'mbti': user_profile.get('mbtiOption', ""),
            'contactFrequencyOption': user_profile.get('contactfrequencyOption', ""),
            'bigHobby': user_profile.get('bigHobbyOption', ""),
            'ageOption': user_profile.get('ageOption', "")  # ✅ 추가
        }
        user_text = build_weighted_text_for_row(
            pseudo_row, mbti_weight, contact_weight, hobby_weight, age_weight  # ✅ 수정
        )

        # 2) TF-IDF 변환
        user_vec = data['vectorizer'].transform([user_text])
        cos_sim = cosine_similarity(user_vec, data['tfidf_matrix'])
        sim_scores = cos_sim[0]

        # 3) 상위 top_k 추천
        sorted_indices = np.argsort(-sim_scores)
        top_indices = sorted_indices[:top_k]

        recommendations = []
        for idx in top_indices:
            row = data['df'].iloc[idx]
            rec = {
                'uuid': row.get('uuid', None),
                'score': float(sim_scores[idx]),
                'gender': row.get('gender', None),
                'mbti': row.get('mbti', ""),
                'age': row.get('age', ""),
                'contactFrequencyOption': row.get('contactFrequencyOption', ""),
                'Hobby': row.get('hobby', ""),
                'major': row.get('major', ""),
            }
            recommendations.append(rec)

        return recommendations
