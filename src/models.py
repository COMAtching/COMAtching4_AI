import pandas as pd
import numpy as np
import argparse  # ✅ CLI 실행 지원 추가
import json  # ✅ 변환된 데이터를 JSON으로 변환
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

# ✅ 현재 디렉토리를 sys.path에 추가하여 패키지 문제 해결
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functions import build_weighted_text_for_row  # ✅ 상대 경로 (.) 삭제하여 절대 경로로 변경

class CosineSimilarityRecommender:
    """
    MBTI + 연락빈도 + 대분류 취미 기반 코사인 유사도 추천 모델
    """
    def __init__(self):
        pass

    def recommend(self, user_profile, data, mbti_weight, contact_weight, hobby_weight, top_k=5):
        """
        user_profile: {'mbtiOption': ..., 'contactfrequencyOption': ..., 'bigHobbyOption': ...}
        data: {
          'df': 필터링된 DataFrame,
          'tfidf_matrix': ...,
          'vectorizer': ...
        }
        """
        if data is None:
            return []

            # ✅ df가 딕셔너리라면 Pandas DataFrame으로 변환
        if isinstance(data['df'], dict):
            data['df'] = pd.DataFrame(data['df'])

        if data['df'].empty:
            return []

        # 1) 유저 weighted text 생성
        pseudo_row = {
            'mbti': user_profile.get('mbtiOption', ""),
            'contactFrequencyOption': user_profile.get('contactfrequencyOption', ""),
            'bigHobby': user_profile.get('bigHobbyOption', "")
        }
        user_text = build_weighted_text_for_row(
            pseudo_row, mbti_weight, contact_weight, hobby_weight
        )

        # 2) TF-IDF 변환
        user_vec = data['vectorizer'].transform([user_text])
        cos_sim = cosine_similarity(user_vec, data['tfidf_matrix'])
        sim_scores = cos_sim[0]  # shape (N, )

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


class ItemBasedCF:
    """
    Item-based Collaborative Filtering 추천 모델
    """
    def __init__(self):
        pass

    def recommend(self, user_profile, pivot_table, top_n=1):
        """
        pivot_table: user_id x item 형태의 매트릭스
        """
        if pivot_table is None or pivot_table.empty:
            return []

        target_user_uuid = user_profile.get('matcherUuid', "unknown_user")
        if target_user_uuid not in pivot_table.index:
            return []

        user_vector = pivot_table.loc[target_user_uuid].values

        # 아이템-아이템 유사도 계산
        item_sim = cosine_similarity(pivot_table.T)
        np.fill_diagonal(item_sim, 0)

        # 사용자 취미와 유사도 점수 예측
        user_scores = item_sim.dot(user_vector)

        # 이미 가지고 있는 취미 제외
        user_scores[user_vector > 0] = -1

        # 정렬 후 상위 추천
        recommended_item_indices = np.argsort(-user_scores)
        recommendations = []
        for idx in recommended_item_indices[:top_n]:
            if user_scores[idx] <= 0:
                break
            item_name = pivot_table.columns[idx]
            recommendations.append({
                'item': item_name,
                'score': user_scores[idx]
            })

        return recommendations


class NeuralCF:
    """
    Neural Collaborative Filtering 추천 모델 (구조 예시)
    """
    def __init__(self):
        pass

    def recommend(self, user_profile, data, top_n=5):
        """
        data['df']: user_id, item_id, rating
        data['user_to_idx']: 사용자 -> index 매핑
        data['hobby_to_idx']: 취미 -> index 매핑
        """
        if data is None:
            return []

        df_ncf = data['df']
        user_to_idx = data['user_to_idx']
        hobby_to_idx = data['hobby_to_idx']

        target_user_uuid = user_profile.get('matcherUuid', "")
        if target_user_uuid not in user_to_idx:
            return []

        target_user_id = user_to_idx[target_user_uuid]

        # 임의로 점수 예측 (실제론 학습된 모델 필요)
        user_had_items = df_ncf[df_ncf['user_id'] == target_user_id]['item_id'].unique().tolist()

        scores = []
        for item, item_id in hobby_to_idx.items():
            if item_id not in user_had_items:
                pred_score = np.random.uniform(0.5, 1.0)
                scores.append((item, pred_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        recommended = [
            {'item': s[0], 'score': s[1]}
            for s in scores[:top_n]
        ]

        return recommended


# ✅ CLI에서 실행 가능하도록 argparse 추가
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="추천 시스템 실행")
    parser.add_argument("--model", type=str, required=True, choices=["cosine", "item_cf", "neural_cf"], help="추천 모델 선택")
    parser.add_argument("--profile", type=str, required=True, help="사용자 프로필 JSON 입력")
    parser.add_argument("--data", type=str, required=True, help="데이터 JSON 입력")
    parser.add_argument("--mbti_weight", type=float, default=1.0, help="MBTI 가중치")
    parser.add_argument("--contact_weight", type=float, default=1.0, help="연락 빈도 가중치")
    parser.add_argument("--hobby_weight", type=float, default=1.0, help="취미 가중치")
    parser.add_argument("--top_n", type=int, default=5, help="추천 개수")

    args = parser.parse_args()

    # JSON 입력값 파싱
    user_profile = json.loads(args.profile)
    data = json.loads(args.data)

    if args.model == "cosine":
        recommender = CosineSimilarityRecommender()
        recommendations = recommender.recommend(
            user_profile, data, args.mbti_weight, args.contact_weight, args.hobby_weight, args.top_n
        )
    elif args.model == "item_cf":
        recommender = ItemBasedCF()
        recommendations = recommender.recommend(user_profile, data, args.top_n)
    elif args.model == "neural_cf":
        recommender = NeuralCF()
        recommendations = recommender.recommend(user_profile, data, args.top_n)

    # ✅ 변환된 데이터 출력 추가
    print("\n✅ 추천 결과:")
    print(json.dumps(recommendations, indent=4, ensure_ascii=False))