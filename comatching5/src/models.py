import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .functions import build_weighted_text_for_row

class CosineSimilarityRecommender:
    """
    MBTI + 연락빈도 + 대분류 취미 기반 코사인 유사도
    """
    def __init__(self):
        pass

    def recommend(self, user_profile, data, mbti_weight, contact_weight, hobby_weight, top_k=5):
        """
        data: {
          'df': 필터링된 DataFrame,
          'tfidf_matrix': ...,
          'vectorizer': ...
        }
        user_profile: {'mbtiOption': ..., 'contactfrequencyOption': ..., 'bigHobbyOption': ...}
        """
        if data is None or data['df'].empty:
            return []

        # 1) 유저 weighted text
        pseudo_row = {
            'mbti': user_profile.get('mbtiOption', ""),
            'contactFrequency': user_profile.get('contactfrequencyOption', ""),
            'bigHobby': user_profile.get('bigHobbyOption', "")
        }
        user_text = build_weighted_text_for_row(
            pseudo_row, mbti_weight, contact_weight, hobby_weight
        )

        # 2) TF-IDF transform
        user_vec = data['vectorizer'].transform([user_text])
        cos_sim = cosine_similarity(user_vec, data['tfidf_matrix'])
        sim_scores = cos_sim[0]  # shape (N, )

        # 3) 상위 top_k
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
                'contactFrequency': row.get('contactFrequency', ""),
                'Hobby': row.get('hobby', ""),
                'major': row.get('major', ""),
            }
            recommendations.append(rec)

        return recommendations



class ItemBasedCF:
    """
    Item-based Collaborative Filtering 모델 (간단 예시)
    """
    def __init__(self):
        pass

    def recommend(self, user_profile, pivot_table, top_n=1):
        """
        pivot_table: user_id x item 형태의 매트릭스
        """
        if pivot_table is None or pivot_table.empty:
            return []

        # 만약 user_profile 내부에 uuidOption이 있다고 가정
        # 실제론 뽑는 사람에게서 '하나의 user_id'를 결정해야 함
        target_user_uuid = user_profile.get('matcherUuid', "unknown_user")
        if target_user_uuid not in pivot_table.index:
            # 만약 대상 user_id가 pivot_table에 없으면 빈 리스트
            return []

        # 사용자가 이미 가지고 있는 취미들
        user_vector = pivot_table.loc[target_user_uuid].values

        # 아이템 기반: 아이템-아이템 유사도 계산
        # 간단하게 코사인 유사도 사용
        item_sim = cosine_similarity(pivot_table.T)  # 아이템 기준
        np.fill_diagonal(item_sim, 0)

        # 사용자 취미(점수가 있는 항목)와 아이템 유사도 점수를 통해 점수 예측
        # 여기서는 단순히 "사용자가 1점 준 아이템"의 유사도 합을 통해 스코어 계산
        user_scores = item_sim.dot(user_vector)

        # 이미 사용자가 1점을 준 아이템(=취미)은 제외
        user_scores[user_vector > 0] = -1

        # 정렬
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
    Neural Collaborative Filtering 모델 (간단 예시)
    실제로는 PyTorch나 TensorFlow 등을 이용해 Embedding 학습 과정을 거쳐야 함.
    여기서는 구조만 간단히 시연.
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

        # 여기서는 임의로 (user_id, item_id) 예측 점수를 = 1 로 가정
        # 실제론 학습된 모델의 예측값이 필요
        # 모든 취미 item_id에 대해 점수 계산 -> 사용자가 가진 취미는 제외
        user_had_items = df_ncf[df_ncf['user_id'] == target_user_id]['item_id'].unique().tolist()

        scores = []
        for item, item_id in hobby_to_idx.items():
            if item_id not in user_had_items:
                # 임의로 0.5 ~ 1.0 범위의 랜덤 스코어
                pred_score = np.random.uniform(0.5, 1.0)
                scores.append((item, pred_score))

        # 정렬
        scores.sort(key=lambda x: x[1], reverse=True)
        recommended = [
            {'item': s[0], 'score': s[1]}
            for s in scores[:top_n]
        ]

        return recommended
