import os
import sys
import time
import uuid
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

# 데이터 경로 설정 (엑셀 파일)
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "comatching_ai_csv1.csv")


def main():
    total_start = time.time()

    # 1) argparse 인자 파싱: uuid, 소분류, 가중치(m, c, h)
    t0 = time.time()
    args = get_args()
    user_uuid = args.uuid if args.uuid is not None else ""
    subcategory = args.subcategory if args.subcategory is not None else ""
    mbti_weight = args.m
    contact_weight = args.c
    hobby_weight = args.h
    t1 = time.time()
    print(f"[1] argparse 파싱 소요 시간: {t1 - t0:.4f}초")

    # 2) 소분류 문자열에 쉼표가 포함되어 있으면 분리하여 리스트로 처리
    if "," in subcategory:
        subcategories = [s.strip() for s in subcategory.split(",")]
    else:
        subcategories = [subcategory]

    # ───────── GPT 분류기 준비 ─────────
    gpt = GPTClassifier(api_key=API_KEY) if API_KEY else None  # ★ 여기 추가/변경

    # 3) 대분류 매핑
    t0 = time.time()
    if len(subcategories) > 1:
        big_category = classify_categories(user_uuid, subcategories, gpt)  # ← gpt
    else:
        big_category = classify_category(user_uuid, subcategories[0], gpt)  # ← gpt
    uuid_val = user_uuid
    t1 = time.time()
    print(f"[2] 대분류 매핑 소요 시간: {t1 - t0:.4f}초")

    # 4) 엑셀 파일 불러오기 (후보 데이터)
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, header=None)
    t1 = time.time()
    print(f"[3] 엑셀 불러오기 소요 시간: {t1 - t0:.4f}초")

    # 5) 유저 프로필 생성 (엑셀 데이터 기반; hobbyOption은 BE에서 받은 소분류 사용)
    t0 = time.time()
    user_profile = create_user_profile(df)
    user_profile["hobbyOption"] = subcategory
    # 대분류 정보도 함께 저장 (추천 결과 해석에 활용)
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
        hobby_weight=hobby_weight
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
