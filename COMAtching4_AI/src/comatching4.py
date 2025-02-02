# src/COMAtching4_AI.py

import os
import time
import pandas as pd

from .args import get_args
from .functions import (
    create_user_profile,
    filter_data,
    preprocess_hobby,
    preprocess_for_cosine,
    # 필요하다면 convert_hobbies_keep_unmatched 등 추가
)
from .agent import manager_agent
from .models import CosineSimilarityRecommender

DATA_PATH = os.path.join('.', 'data', 'comatching_ai_csv.xlsx')

def parse_big_category_result(gpt_response: str) -> dict:
    """
    GPT 응답을 파싱하여 {대분류: 등장횟수} 형태로 변환
    """
    result_dict = {}
    segments = gpt_response.split(',')
    for seg in segments:
        seg = seg.strip()
        if '(' in seg and ')' in seg:
            cat = seg.split('(')[0].strip()
            cnt_str = seg.split('(')[1].replace(')', '').strip()
            try:
                cnt = int(cnt_str)
            except:
                cnt = 1
            result_dict[cat] = result_dict.get(cat, 0) + cnt
        else:
            if seg:
                result_dict[seg] = result_dict.get(seg, 0) + 1
    return result_dict

def main():
    total_start = time.time()

    # 1) argparse 인자 파싱
    t0 = time.time()
    args = get_args()
    mbti_weight = args.m
    contact_weight = args.c
    hobby_weight = args.h
    t1 = time.time()
    print(f"[1] argparse 파싱 소요 시간: {t1 - t0:.4f}초")

    # 2) 엑셀 불러오기
    t0 = time.time()
    df = pd.read_excel(DATA_PATH, header=None)
    t1 = time.time()
    print(f"[2] 엑셀 불러오기 소요 시간: {t1 - t0:.4f}초")

    # 3) 유저 프로필 생성
    t0 = time.time()
    user_profile = create_user_profile(df)
    t1 = time.time()
    print(f"[3] 유저 프로필 생성 소요 시간: {t1 - t0:.4f}초")

    # 4) 필터링
    t0 = time.time()
    filtered_df = filter_data(df, user_profile)
    t1 = time.time()
    print(f"[4] 필터링 소요 시간: {t1 - t0:.4f}초")

    # (Optional) 4-1) 뽑히는 사람 / 뽑는 사람의 소분류 -> 대분류 변환
    # t0 = time.time()
    # filtered_df, user_profile = convert_hobbies_keep_unmatched(filtered_df, user_profile)
    # t1 = time.time()
    # print(f"[4-1] 소분류 → 대분류 변환 소요 시간: {t1 - t0:.4f}초")

    # 5) 뽑는 사람 소분류 취미 추출 -> GPT agent에 넘길 준비
    t0 = time.time()
    sub_hobbies = preprocess_hobby(user_profile)
    t1 = time.time()
    print(f"[5] 소분류 취미 추출 소요 시간: {t1 - t0:.4f}초")

    # 6) GPT로 분류 (소분류 -> 대분류)
    t0 = time.time()
    gpt_response = manager_agent(
        extracted_info=sub_hobbies,
        user_profile=user_profile,
        extractor_message="추가 분류 작업을 진행해주세요."
    )
    t1 = time.time()
    print(f"[6] GPT 에이전트 호출 소요 시간: {t1 - t0:.4f}초")

    # 7) GPT 응답 파싱
    t0 = time.time()
    big_cat_dict = parse_big_category_result(gpt_response)
    big_hobby_list = []
    for cat, cnt in big_cat_dict.items():
        for _ in range(cnt):
            big_hobby_list.append(cat)
    user_profile['bigHobbyOption'] = " ".join(big_hobby_list)
    t1 = time.time()
    print(f"[7] GPT 응답 파싱 + bigHobbyOption 구성 소요 시간: {t1 - t0:.4f}초")

    # 8) 코사인 유사도용 전처리
    t0 = time.time()
    cosine_data = preprocess_for_cosine(
        filtered_df,
        mbti_weight=mbti_weight,
        contact_weight=contact_weight,
        hobby_weight=hobby_weight
    )
    t1 = time.time()
    print(f"[8] 코사인 유사도 전처리 소요 시간: {t1 - t0:.4f}초")

    # 9) 추천 모델
    t0 = time.time()
    cosine_model = CosineSimilarityRecommender()
    recommendations = cosine_model.recommend(
        user_profile,
        data=cosine_data,
        mbti_weight=mbti_weight,
        contact_weight=contact_weight,
        hobby_weight=hobby_weight,
        top_k=1
    )
    t1 = time.time()
    print(f"[9] 모델 추천 소요 시간: {t1 - t0:.4f}초")

    # 결과 출력
    print("\n===== Cosine Similarity 추천 결과 =====")
    for rec in recommendations:
        print(rec)

    total_end = time.time()
    print(f"\n[총 실행 시간] {total_end - total_start:.4f}초")

if __name__ == "__main__":
    main()
