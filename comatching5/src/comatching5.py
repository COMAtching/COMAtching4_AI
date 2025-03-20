#
# import os
# import sys
# import time
# import pandas as pd
# from dotenv import load_dotenv
#
# # ✅ 현재 디렉토리를 sys.path에 추가하여 패키지 인식 문제 해결
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#
# from args import get_args
# from functions import (
#     create_user_profile,
#     filter_data,
#     preprocess_for_cosine,
# )
# from agent import manager_agent  # ✅ 상대 경로 (.) 삭제하여 절대 경로로 변경
# from models import CosineSimilarityRecommender
# from utils import extract_sub_hobbies, parse_gpt_response, GPTClassifier
#
# # 🔹 환경 변수에서 API 키 로드
# load_dotenv()
# API_KEY = os.getenv("OPENAI_API_KEY")
#
# # DATA_PATH = os.path.abspath(os.path.join('data', 'comatching_ai_csv.xlsx'))
# DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "comatching_ai_csv.xlsx")
# def main():
#     total_start = time.time()
#
#     # 1) argparse 인자 파싱
#     t0 = time.time()
#     args = get_args()
#     mbti_weight = args.m
#     contact_weight = args.c
#     hobby_weight = args.h
#     subcategory = args.subcategory
#     t1 = time.time()
#     print(f"[1] argparse 파싱 소요 시간: {t1 - t0:.4f}초" )
#
#     # 2) 엑셀 불러오기
#     t0 = time.time()
#     df = pd.read_excel(DATA_PATH, header=None)
#     t1 = time.time()
#     print(f"[2] 엑셀 불러오기 소요 시간: {t1 - t0:.4f}초")
#
#     # 3) 유저 프로필 생성
#     t0 = time.time()
#     user_profile = create_user_profile(df)
#     # ✅ 사용자가 입력한 `--subcategory` 값 반영
#     if subcategory:
#         user_profile["hobbyOption"] = subcategory
#     t1 = time.time()
#     print(f"[3] 유저 프로필 생성 소요 시간: {t1 - t0:.4f}초")
#
#     # 4) 필터링
#     t0 = time.time()
#     filtered_df = filter_data(df, user_profile)
#     t1 = time.time()
#     print(f"[4] 필터링 소요 시간: {t1 - t0:.4f}초")
#
#     # (Optional) 4-1) 뽑히는 사람 / 뽑는 사람의 소분류 -> 대분류 변환
#
#     # 5) 뽑는 사람 소분류 취미 추출 -> GPT agent에 넘길 준비
#     t0 = time.time()
#     sub_hobbies = extract_sub_hobbies(user_profile)
#     t1 = time.time()
#     print(f"[5] 소분류 취미 추출 소요 시간: {t1 - t0:.4f}초")
#
#     # 6) GPT로 분류 (소분류 -> 대분류)
#     t0 = time.time()
#     gpt_classifier = GPTClassifier(api_key=API_KEY)
#     gpt_response = gpt_classifier.classify_hobbies(sub_hobbies, user_profile)
#     t1 = time.time()
#     print(f"[6] GPT 에이전트 호출 소요 시간: {t1 - t0:.4f}초")
#
#     # ✅ 변경된 부분: gpt_response가 딕셔너리이므로 "bigcategory" 값만 사용
#     big_category_text = gpt_response.get("bigcategory", "")
#
#     # 7) GPT 응답 파싱
#     t0 = time.time()
#     big_cat_dict = parse_gpt_response(big_category_text)  # ✅ bigcategory 값만 넘김
#     big_hobby_list = [cat for cat, cnt in big_cat_dict.items() for _ in range(cnt)]
#     user_profile['bigHobbyOption'] = " ".join(big_hobby_list)
#     t1 = time.time()
#     print(f"[7] GPT 응답 파싱 + bigHobbyOption 구성 소요 시간: {t1 - t0:.4f}초")
#
#     # 8) 코사인 유사도용 전처리
#     t0 = time.time()
#     cosine_data = preprocess_for_cosine(
#         filtered_df,
#         mbti_weight=mbti_weight,
#         contact_weight=contact_weight,
#         hobby_weight=hobby_weight
#     )
#     t1 = time.time()
#     print(f"[8] 코사인 유사도 전처리 소요 시간: {t1 - t0:.4f}초")
#
#     # 9) 추천 모델
#     t0 = time.time()
#     cosine_model = CosineSimilarityRecommender()
#     recommendations = cosine_model.recommend(
#         user_profile,
#         data=cosine_data,
#         mbti_weight=mbti_weight,
#         contact_weight=contact_weight,
#         hobby_weight=hobby_weight,
#         top_k=1  # 추천 인원 수
#     )
#     t1 = time.time()
#     print(f"[9] 모델 추천 소요 시간: {t1 - t0:.4f}초")
#
#     # 결과 출력 (리스트 형태)
#     print("\n===== Cosine Similarity 추천 결과 =====")
#     for idx, rec in enumerate(recommendations, 1):
#         print(f"[{idx}] 성별: {rec['gender']}, MBTI: {rec['mbti']}, 나이: {rec['age']}, 연락 빈도: {rec['contactFrequencyOption']}")
#         print(f"   취미: {rec['Hobby']}, 전공: {rec['major']}, 추천 점수: {rec['score']}")
#         print("-" * 50)  # 가독성을 위한 구분선 추가
#
#     total_end = time.time()
#     print(f"\n[총 실행 시간] {total_end - total_start:.4f}초")
#
# if __name__ == "__main__":
#     main()

import os
import sys
import time
import uuid
import pandas as pd
from dotenv import load_dotenv

# ✅ 현재 디렉토리를 sys.path에 추가하여 패키지 인식 문제 해결
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from args import get_args
from functions import (
    create_user_profile,
    filter_data,
    preprocess_for_cosine,
)
from agent import manager_agent  # ✅ 상대 경로 (.) 삭제하여 절대 경로로 변경
from models import CosineSimilarityRecommender
from utils import extract_sub_hobbies, parse_gpt_response, GPTClassifier

# 🔹 환경 변수에서 API 키 로드
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# DATA_PATH = os.path.abspath(os.path.join('data', 'comatching_ai_csv.xlsx'))
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "comatching_ai_csv.xlsx")
def main():
    total_start = time.time()

    # 1) argparse 인자 파싱
    t0 = time.time()
    args = get_args()
    mbti_weight = args.m
    contact_weight = args.c
    hobby_weight = args.h
    subcategory = args.subcategory
    t1 = time.time()
    print(f"[1] argparse 파싱 소요 시간: {t1 - t0:.4f}초" )

    # 2) 엑셀 불러오기
    t0 = time.time()
    df = pd.read_excel(DATA_PATH, header=None)
    t1 = time.time()
    print(f"[2] 엑셀 불러오기 소요 시간: {t1 - t0:.4f}초")

    # 3) 유저 프로필 생성
    t0 = time.time()
    user_profile = create_user_profile(df)
    # ✅ 사용자가 입력한 `--subcategory` 값 반영
    if subcategory:
        user_profile["hobbyOption"] = subcategory
    t1 = time.time()
    print(f"[3] 유저 프로필 생성 소요 시간: {t1 - t0:.4f}초")

    # 4) 필터링
    t0 = time.time()
    filtered_df = filter_data(df, user_profile)
    t1 = time.time()
    print(f"[4] 필터링 소요 시간: {t1 - t0:.4f}초")

    # (Optional) 4-1) 뽑히는 사람 / 뽑는 사람의 소분류 -> 대분류 변환

    # 5) 뽑는 사람 소분류 취미 추출 -> GPT agent에 넘길 준비
    t0 = time.time()
    sub_hobbies = extract_sub_hobbies(user_profile)
    t1 = time.time()
    print(f"[5] 소분류 취미 추출 소요 시간: {t1 - t0:.4f}초")

    # 6) GPT로 분류 (소분류 -> 대분류)
    t0 = time.time()
    gpt_classifier = GPTClassifier(api_key=API_KEY)
    gpt_response = gpt_classifier.classify_hobbies(sub_hobbies, user_profile)
    t1 = time.time()
    print(f"[6] GPT 에이전트 호출 소요 시간: {t1 - t0:.4f}초")

    # ✅ 변경된 부분: gpt_response가 딕셔너리이므로 "bigcategory" 값만 사용
    big_category_text = gpt_response.get("bigcategory", "")

    # 7) GPT 응답 파싱
    t0 = time.time()
    big_cat_dict = parse_gpt_response(big_category_text)  # ✅ bigcategory 값만 넘김
    big_hobby_list = [cat for cat, cnt in big_cat_dict.items() for _ in range(cnt)]
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
        top_k=1  # 추천 인원 수
    )
    t1 = time.time()
    print(f"[9] 모델 추천 소요 시간: {t1 - t0:.4f}초")

    # 결과 출력 (리스트 형태)
    print("\n===== Cosine Similarity 추천 결과 =====")
    for idx, rec in enumerate(recommendations, 1):
        rec_uuid = str(uuid.uuid4())  # UUID 생성
        print(f"[{idx}] uuid: {rec_uuid}")
        print(f"   성별: {rec['gender']}, MBTI: {rec['mbti']}, 나이: {rec['age']}, 연락 빈도: {rec['contactFrequencyOption']}")
        print(f"   취미: {rec['Hobby']}, 전공: {rec['major']}, 추천 점수: {rec['score']}")
        print("-" * 50)  # 가독성을 위한 구분선 추가

    total_end = time.time()
    print(f"\n[총 실행 시간] {total_end - total_start:.4f}초")

if __name__ == "__main__":
    main()