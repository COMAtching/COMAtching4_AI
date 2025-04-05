# comatching_category_classification.py

import argparse
import os
from dotenv import load_dotenv
from utils import parse_gpt_response, GPTClassifier

# 대분류 - 소분류 매핑 딕셔너리
BIG_CATEGORY_DICT = {
    "스포츠": [
        "축구", "농구", "야구", "배드민턴", "헬스",
        "필라테스", "클라이밍", "자전거", "러닝"
    ],
    "문화": [
        "독서", "영화", "웹툰", "OTT 시청", "방탈출",
        "뮤지컬", "전시회", "페스티벌"
    ],
    "예술": [
        "인디음악", "힙합", "발라드", "k-pop", "해외팝송",
        "악기 연주", "댄스", "패션", "사진 촬영"
    ],
    "여행": [
        "국내 여행", "해외 여행", "맛집 탐방", "캠핑", "등산", "드라이브"
    ],
    "자기계발": [
        "요리", "베이킹", "수집", "뜨개질", "티",
        "투자", "휴식", "코딩"
    ],
    "게임": [
        "보드게임", "롤", "롤토체스", "피파", "배그",
        "오버워치", "스팀게임", "모바일게임"
    ]
}


def get_big_category_from_dict(subcategory):
    """
    소분류(subcategory)가 딕셔너리 내에 있으면 해당 대분류를 반환합니다.
    """
    for big_category, sub_list in BIG_CATEGORY_DICT.items():
        if subcategory in sub_list:
            return big_category
    return None


def classify_category(uuid, subcategory, api_key):
    """
    BE에서 전달받은 uuid와 단일 소분류를 기반으로 대분류를 결정합니다.
      - 소분류가 BIG_CATEGORY_DICT에 있으면 그대로 대분류로 사용합니다.
      - 없으면 GPT API를 호출하여 대분류를 결정합니다.
    최종적으로 uuid와 대분류를 반환합니다.
    """
    # 먼저 딕셔너리에서 대분류를 찾습니다.
    big_category = get_big_category_from_dict(subcategory) if subcategory else ""

    # 딕셔너리에 없으면 GPT API 호출
    if not big_category and subcategory:
        gpt_classifier = GPTClassifier(api_key=api_key)
        # 소분류를 리스트로 전달합니다.
        gpt_response = gpt_classifier.classify_hobbies([subcategory], {"uuid": uuid, "hobbyOption": subcategory})
        # GPT 응답에서 "bigcategory" 키로 대분류 결과를 추출합니다.
        big_category_text = gpt_response.get("bigcategory", "")
        big_cat_dict = parse_gpt_response(big_category_text)
        if big_cat_dict:
            # 결과가 여러 개라면 첫 번째 키를 사용합니다.
            big_category = list(big_cat_dict.keys())[0]
        else:
            big_category = ""
    return uuid, big_category


def classify_categories(uuid, subcategories, api_key):
    """
    여러 소분류에 대해 각각 대분류를 결정한 후,
    중복 제거하여 콤마(,)로 구분된 문자열로 반환합니다.
    """
    big_categories = []
    for sub in subcategories:
        _, big_cat = classify_category(uuid, sub, api_key)
        if big_cat and big_cat not in big_categories:
            big_categories.append(big_cat)
    return uuid, ", ".join(big_categories)


def main():
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")

    parser = argparse.ArgumentParser(
        description="단독 실행 시 uuid와 소분류를 기반으로 대분류를 매핑합니다."
    )
    # 여러 소분류를 입력받기 위해 nargs='+' 사용 (공백으로 구분)
    parser.add_argument('--uuid', type=str, default="default_uuid", help="사용자 UUID (기본: default_uuid)")
    parser.add_argument('--subcategory', type=str, nargs='+', default=["축구"], help="소분류 데이터 (기본: 축구)")
    args = parser.parse_args()

    # 입력받은 소분류가 하나의 요소이고, 해당 요소에 콤마가 포함되어 있다면 분리 처리
    subcategories = args.subcategory
    if len(subcategories) == 1 and "," in subcategories[0]:
        subcategories = [s.strip() for s in subcategories[0].split(",")]

    if len(subcategories) > 1:
        uuid_val, big_category = classify_categories(args.uuid, subcategories, API_KEY)
    else:
        uuid_val, big_category = classify_category(args.uuid, subcategories[0], API_KEY)

    print("UUID:", uuid_val)
    print("대분류:", big_category)


if __name__ == "__main__":
    main()
