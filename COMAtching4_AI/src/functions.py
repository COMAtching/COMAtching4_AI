import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def recommendation_function(user_profile, file_path, embedding_model, user_id):
    try:
        # CSV 파일 읽기
        df = pd.read_excel(file_path)

        # 필수 필드 확인
        if 'mbti' not in df.columns:
            raise ValueError("데이터셋에 'mbti' 필드가 없습니다.")

        # 데이터 전처리: mbti 값 확인 및 처리
        missing_mbti = df[df['mbti'].isna()]
        if not missing_mbti.empty:
            print(f"다음 행은 'mbti' 값이 누락되어 제거됩니다: {missing_mbti}")
        df = df.dropna(subset=['mbti'])
        df['mbti'] = df['mbti'].astype(str).str.strip()

        # 본인 제외
        df = df[df['id'].astype(str) != user_id]

        # 성별 필터링
        user_gender = user_profile.get('gender')
        if user_gender == '남':
            df = df[df['gender'] == '여']
        elif user_gender == '여':
            df = df[df['gender'] == '남']
        else:
            raise ValueError("사용자의 성별 정보가 올바르지 않습니다.")

        # 나이 선호 필터링
        user_age = user_profile.get('age')
        age_preference = user_profile.get('agePreference')
        if age_preference == '연상':
            df = df[df['age'] > user_age]
        elif age_preference == '동갑':
            df = df[df['age'] == user_age]
        elif age_preference == '연하':
            df = df[df['age'] < user_age]

        # 후보자들도 필수 필드가 있어야 함
        df = df.dropna(subset=['contactFrequency', 'hobby'])

        # 사용자 프로필에서 비교할 텍스트 생성
        user_text = f"{user_profile['contactFrequency']} {user_profile['hobby']} {user_profile.get('mbtiPreference', '')}"

        # 후보자들의 텍스트 생성
        df['candidate_text'] = df.apply(
            lambda row: f"{row['contactFrequency']} {row['hobby']} {row['mbti']}", axis=1)

        # 사용자와 후보자들의 임베딩 생성
        user_embedding = embedding_model.encode(user_text, convert_to_tensor=True)
        candidate_embeddings = embedding_model.encode(df['candidate_text'].tolist(), convert_to_tensor=True)

        # 코사인 유사도 계산
        cosine_scores = cosine_similarity(
            user_embedding.cpu().numpy().reshape(1, -1),
            candidate_embeddings.cpu().numpy()
        )[0]

        df['similarity'] = cosine_scores

        # 유사도 기반 정렬 후 최적의 후보 반환
        if not df.empty:
            best_match = df.sort_values(by='similarity', ascending=False).iloc[0]
            return best_match

        print("추천 결과를 찾을 수 없습니다.")
        return None

    except Exception as e:
        print(f"추천 과정에서 오류 발생: {e}")
        return None


def deliver_recommendation(recommended_person, conversation_history):
    """
    추천 결과를 사용자에게 전달
    """
    if recommended_person is None:
        return "추천 결과를 생성할 수 없습니다. 입력 정보를 확인해주세요."

    recommendation_message = f"""
내가 생각해봤는데, 너에게 딱 맞는 친구를 찾아봤어!

id: {recommended_person['id']}
나이: {recommended_person['age']}
연락 빈도: {recommended_person['contactFrequency']}
취미: {recommended_person['hobby']}
MBTI: {recommended_person['mbti']}
음악: {recommended_person['music']}
한줄소개: {recommended_person['description']}
"""
    conversation_history.append({"role": "assistant", "content": recommendation_message})
    return recommendation_message
