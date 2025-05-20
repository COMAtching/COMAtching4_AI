import os
import sys
import time
import pandas as pd
from dotenv import load_dotenv

# 현재 디렉토리를 sys.path에 추가하여 모듈 인식 문제 해결
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from args import get_args
from functions import create_user_profile, filter_data, preprocess_for_cosine
from models import CosineSimilarityRecommender
from comatching_category_classification import classify_category, classify_categories
from utils import GPTClassifier

# 환경 변수에서 API 키 로드
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# 데이터 경로 설정 (CSV 파일)
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "comatching_ai_csv1.csv")

def main():
    total_start = time.time()

    # 1) argparse 인자 파싱: uuid, 소분류
    t0 = time.time()
    args = get_args()
    user_uuid = args.uuid if args.uuid is not None else ""
    subcategory = args.subcategory if args.subcategory is not None else ""
    t1 = time.time()
    print(f"[1] argparse 파싱 소요 시간: {t1 - t0:.4f}초")

    # 2) 소분류 문자열에 쉼표가 포함되어 있으면 분리하여 리스트로 처리
    if "," in subcategory:
        subcategories = [s.strip() for s in subcategory.split(",")]
    else:
        subcategories = [subcategory]

    # ───────── GPT 분류기 준비 ─────────
    gpt = GPTClassifier(api_key=API_KEY) if API_KEY else None

    # 3) 대분류 매핑
    t0 = time.time()
    if len(subcategories) > 1:
        big_category = classify_categories(user_uuid, subcategories, gpt)
    else:
        big_category = classify_category(user_uuid, subcategories[0], gpt)
    uuid_val = user_uuid
    t1 = time.time()
    print(f"[2] 대분류 매핑 소요 시간: {t1 - t0:.4f}초")

    # 4) CSV 파일 불러오기 및 가중치 추출
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, header=None)  # 헤더가 없다고 가정
    # 가중치 추출: 행 4 (인덱스 3), 열 I, J, K, L (인덱스 8~11)
    mbti_weight = float(df.iloc[3, 8])      # 예: 0.3
    contact_weight = float(df.iloc[3, 9])   # 예: 0.2
    hobby_weight = float(df.iloc[3, 10])    # 예: 0.5
    age_weight = float(df.iloc[3, 11])      # ✅ 추가

    # 가중치 출력
    print(f"MBTI Weight: {mbti_weight}")
    print(f"Contact Weight: {contact_weight}")
    print(f"Hobby Weight: {hobby_weight}")
    print(f"Age Weight: {age_weight}")

    t1 = time.time()
    print(f"[3] CSV 불러오기 및 가중치 추출 소요 시간: {t1 - t0:.4f}초")

    # 5) 유저 프로필 생성 (CSV 데이터 기반; hobbyOption은 BE에서 받은 소분류 사용)
    t0 = time.time()
    user_profile = create_user_profile(df)
    user_profile["hobbyOption"] = subcategory
    user_profile["bigHobbyOption"] = big_category
    t1 = time.time()
    print(f"[4] 유저 프로필 생성 소요 시간: {t1 - t0:.4f}초")

    # 6) 데이터 필터링 (유저 프로필 기준)
    t0 = time.time()
    filtered_df = filter_data(df, user_profile)
    t1 = time.time()
    print(f"[5] 데이터 필터링 소요 시간: {t1 - t0:.4f}초")

    # 7) 코사인 유사도용 전처리
    t0 = time.time()
    cosine_data = preprocess_for_cosine(
        filtered_df,
        mbti_weight=mbti_weight,
        contact_weight=contact_weight,
        hobby_weight=hobby_weight,
        age_weight=age_weight  # ✅ 추가
    )
    t1 = time.time()
    print(f"[6] 코사인 유사도 전처리 소요 시간: {t1 - t0:.4f}초")

    # 8) 추천 모델 실행 (코사인 유사도 기반 추천)
    t0 = time.time()
    cosine_model = CosineSimilarityRecommender()
    recommendations = cosine_model.recommend(
        user_profile,
        data=cosine_data,
        mbti_weight=mbti_weight,
        contact_weight=contact_weight,
        hobby_weight=hobby_weight,
        age_weight=age_weight,  # ✅ 추가
        top_k=1  # 추천 인원 수
    )
    t1 = time.time()
    print(f"[7] 추천 모델 실행 소요 시간: {t1 - t0:.4f}초")

    # 9) 최종 추천 결과 출력
    print("\n===== Cosine Similarity 추천 결과 =====")
    for idx, rec in enumerate(recommendations, 1):
        rec_uuid = rec.get("uuid")
        print(f"[{idx}] uuid: {rec_uuid}")
        print(
            f"   성별: {rec.get('gender', 'N/A')}, MBTI: {rec.get('mbti', 'N/A')}, 나이: {rec.get('age', 'N/A')}, 연락 빈도: {rec.get('contactFrequencyOption', 'N/A')}")
        print(f"   취미: {rec.get('Hobby', 'N/A')}, 전공: {rec.get('major', 'N/A')}, 추천 점수: {rec.get('score', 'N/A')}")
        print("-" * 50)

    total_end = time.time()
    print(f"\n[총 실행 시간] {total_end - total_start:.4f}초")

    # 최종적으로 uuid와 대분류 반환
    return uuid_val, big_category

if __name__ == "__main__":
    main()
