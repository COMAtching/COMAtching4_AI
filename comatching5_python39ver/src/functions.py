# src/functions.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse

"""
1) 유저 프로필 생성 함수
2) 필터링 함수 (duplication='TRUE' 제거, 성별, sameMajorOption, 나이 필터링)
3) 소분류 취미 전처리 함수
4) 코사인 유사도 전처리 함수
"""

# 1. 대분류 - 소분류 매핑 딕셔너리
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


def map_small_to_big(hobby: str, big_dict: dict) -> str:
    for big_cat, small_list in big_dict.items():
        if hobby in small_list:
            return big_cat
    return ""


# -----------------------------------------------------------
# 1) "소분류 → 대분류" + 매핑 불가시 '그대로 사용' (뽑히는 사람)
#    + 뽑는 사람의 취미도 같은 로직으로 시도, 남은 취미들은 GPT에 넘길 수 있음
# -----------------------------------------------------------
def convert_hobbies_keep_unmatched(filtered_df: pd.DataFrame, user_profile: dict) -> (pd.DataFrame, dict):
    """
    1. 뽑히는 사람들의 취미(hobby)를 대분류로 바꿀 수 있으면 바꾸고,
       못 바꾸면(매핑 안 되면) 그대로 남김.
    2. 뽑는 사람의 취미(user_profile['hobbyOption'])도 같은 로직 수행.
       - 바뀌지 않은 소분류는 추후 GPT agent로 분류할 수 있게 남겨둠.
    3. 반환: (새로운 filtered_df, 새로운 user_profile)
    """

    # (A) 뽑히는 사람(필터링된 df) 처리
    if 'hobby' in filtered_df.columns:
        new_hobby_list = []
        for idx, row in filtered_df.iterrows():
            hobby_str = str(row['hobby'])
            split_hobbies = [h.strip() for h in hobby_str.split(',') if h.strip()]
            converted_list = []

            for small_hobby in split_hobbies:
                big_cat = map_small_to_big(small_hobby, BIG_CATEGORY_DICT)
                if big_cat == "":
                    # 매핑 안 되면 '그대로' 사용
                    converted_list.append(small_hobby)
                else:
                    # 대분류로 교체
                    converted_list.append(big_cat)

            # 최종 합치기
            new_hobby_list.append(",".join(converted_list))

        filtered_df = filtered_df.copy()
        filtered_df['hobby'] = new_hobby_list

    # (B) 뽑는 사람(user_profile) 처리
    hobby_option = user_profile.get('hobbyOption', "")
    if hobby_option.strip():
        split_hobbies = [h.strip() for h in hobby_option.split(',') if h.strip()]
        converted_list = []

        for small_hobby in split_hobbies:
            big_cat = map_small_to_big(small_hobby, BIG_CATEGORY_DICT)
            if big_cat == "":
                # 매핑 안 되면 그대로 유지 → GPT agent로 넘길 수 있음
                converted_list.append(small_hobby)
            else:
                # 대분류로 교체
                converted_list.append(big_cat)

        user_profile['hobbyOption'] = ",".join(converted_list)

    return filtered_df, user_profile


# -----------------------------------------------------------
# 2) "소분류 → 대분류" + 매핑 불가시 '잘라서 사용하지 않음' (뽑히는 사람)
#    + 뽑는 사람의 취미도 같은 로직으로 시도, 남은 취미들은 GPT에 넘김
# -----------------------------------------------------------
def convert_hobbies_discard_unmatched(filtered_df: pd.DataFrame, user_profile: dict) -> (pd.DataFrame, dict):
    """
    1. 뽑히는 사람들의 취미(hobby)를 대분류로 바꿀 수 있으면 바꾸고,
       못 바꾸면(매핑 안 되면) '제거'함(즉, 쓰지 않음).
    2. 뽑는 사람의 취미(user_profile['hobbyOption'])도 같은 로직 수행.
       - 바뀌지 않은 소분류는 GPT agent로 넘길 수 있음 (또는 제거할 수도 있음).
    3. 반환: (새로운 filtered_df, 새로운 user_profile)
    """

    # (A) 뽑히는 사람(필터링된 df) 처리
    if 'hobby' in filtered_df.columns:
        new_hobby_list = []
        for idx, row in filtered_df.iterrows():
            hobby_str = str(row['hobby'])
            split_hobbies = [h.strip() for h in hobby_str.split(',') if h.strip()]
            converted_list = []

            for small_hobby in split_hobbies:
                big_cat = map_small_to_big(small_hobby, BIG_CATEGORY_DICT)
                if big_cat == "":
                    # 매핑 안 되면 버림(사용X)
                    continue
                else:
                    converted_list.append(big_cat)

            new_hobby_list.append(",".join(converted_list))

        filtered_df = filtered_df.copy()
        filtered_df['hobby'] = new_hobby_list

    # (B) 뽑는 사람(user_profile) 처리
    hobby_option = user_profile.get('hobbyOption', "")
    if hobby_option.strip():
        split_hobbies = [h.strip() for h in hobby_option.split(',') if h.strip()]
        converted_list = []

        for small_hobby in split_hobbies:
            big_cat = map_small_to_big(small_hobby, BIG_CATEGORY_DICT)
            if big_cat == "":
                continue
            else:
                converted_list.append(big_cat)

        user_profile['hobbyOption'] = ",".join(converted_list)

    return filtered_df, user_profile


def create_user_profile(df: pd.DataFrame) -> dict:
    """
    유저 프로필 생성
    - 0행(인덱스 이름들), 1행(해당 값) 파악 -> user_profile dict
    - MBTI를 4글자로 정제
    - myAge가 존재하고 age가 비어 있다면 age에 복사
    """
    # 0행, 1행
    indices = df.iloc[0].tolist()
    values = df.iloc[1].tolist()

    user_profile = {}
    for i, idx in enumerate(indices):
        val = values[i] if pd.notna(values[i]) else ""
        user_profile[idx] = val

    # MBTI 정제 (콤마 제거 등)
    mbti_raw = user_profile.get('mbtiOption', "")
    if mbti_raw:
        cleaned_mbti = mbti_raw.replace(",", "").replace(" ", "")
        user_profile['mbtiOption'] = cleaned_mbti

    # (추가) myAge → age 매핑
    if user_profile.get('myAge') and not user_profile.get('age'):
        user_profile['age'] = user_profile['myAge']

    return user_profile


def filter_data(df: pd.DataFrame, user_profile: dict) -> pd.DataFrame:
    """
    - 0,1행 삭제
    - duplication=='TRUE' 제거
    - matcherUuid == uuid 제거
    - 성별 필터링
    - sameMajorOption == 'TRUE' 면 같은 학과 전부 제외
    - 나이필터링: ageOption(OLDER/YOUNGER/EQUAL)에 따라 사용자 age보다 크거나/작거나/같은 행만 남김
    """
    # 0, 1행 제거
    filtered_df = df.drop([0, 1], axis=0).reset_index(drop=True)
    # 새 헤더
    new_header = filtered_df.iloc[0]
    filtered_df = filtered_df[1:]
    filtered_df.columns = new_header

    # duplication == 'TRUE' 제거
    filtered_df = filtered_df[filtered_df['duplication'] != 'TRUE']

    # matcherUuid 제거
    matcher_uuid_value = user_profile.get('matcherUuid', "")
    if matcher_uuid_value:
        filtered_df = filtered_df[filtered_df['uuid'] != matcher_uuid_value]

    # 성별 필터링
    user_gender = user_profile.get('genderOption', "")
    if user_gender == 'MALE':
        filtered_df = filtered_df[filtered_df['gender'] == 'FEMALE']
    elif user_gender == 'FEMALE':
        filtered_df = filtered_df[filtered_df['gender'] == 'MALE']

    # sameMajorOption
    same_major_option = user_profile.get('sameMajorOption', "FALSE")
    user_major = user_profile.get('myMajor', "")
    if same_major_option == "TRUE" and user_major:
        filtered_df = filtered_df[filtered_df['major'] != user_major]

    # 나이필터링 (ageOption: OLDER/YOUNGER/EQUAL)
    # age가 없으면 myAge 사용
    user_age_val = user_profile.get('age') or user_profile.get('myAge')
    age_option = user_profile.get('ageOption', "").upper()  # OLDER/YOUNGER/EQUAL

    if user_age_val and str(user_age_val).isdigit():
        user_age_int = int(user_age_val)
        filtered_df['age'] = filtered_df['age'].astype(int)  # 형변환
        if age_option == 'OLDER':
            filtered_df = filtered_df[filtered_df['age'] > user_age_int]
        elif age_option == 'YOUNGER':
            filtered_df = filtered_df[filtered_df['age'] < user_age_int]
        elif age_option == 'EQUAL':
            filtered_df = filtered_df[filtered_df['age'] == user_age_int]

    return filtered_df.reset_index(drop=True)


def preprocess_hobby(user_profile: dict) -> dict:
    """
    유저프로필에서 소분류 취미를 추출 (대분류와 동일한 단어 제외).
    예) "음악감상, 사진" -> {'음악감상':'소분류', '사진':'소분류'}
    """
    big_hobby_categories = {'운동', '음식', '예술', '음악', '테크', '야외활동', '여행', '게임'}
    hobby_option = user_profile.get('hobbyOption', "")

    sub_dict = {}
    if not hobby_option.strip():
        return sub_dict

    split_hobbies = [h.strip() for h in hobby_option.split(',')]
    for h in split_hobbies:
        if h and (h not in big_hobby_categories):
            sub_dict[h] = "소분류"
    return sub_dict


"""
(추가) 코사인 유사도 전처리 관련 함수들
 - 나이는 사용하지 않고, MBTI / 연락빈도 / (GPT 결과) 대분류 취미만 사용
"""


def build_weighted_text_for_row(row, mbtiWeight, contactFrequencyWeight, hobbyWeight, ageWeight):
    text_tokens = []

    if row.get('mbti', None):
        repeat_mbti = int(round(mbtiWeight * 3))
        text_tokens += [row['mbti']] * repeat_mbti

    if row.get('contactFrequencyOption', None):
        repeat_contact = int(round(contactFrequencyWeight * 3))
        text_tokens += [row['contactFrequencyOption']] * repeat_contact

    if row.get('bigHobby', None):
        splitted = row['bigHobby'].split()
        unique_big_cats = set(splitted)
        for token in unique_big_cats:
            repeat_hobby = int(round(hobbyWeight * 3))
            text_tokens += [token] * repeat_hobby

    if row.get('ageOption', None):
        repeat_age = int(round(ageWeight * 3))
        text_tokens += [row['ageOption']] * repeat_age

    return " ".join(text_tokens).strip()


def preprocess_for_cosine(filtered_df: pd.DataFrame,
                          mbtiWeight: float,
                          contactFrequencyWeight: float,
                          hobbyWeight: float,
                          ageWeight: float):
    if filtered_df.empty:
        return None

    if 'bigHobby' not in filtered_df.columns:
        filtered_df['bigHobby'] = ""

    if 'ageOption' not in filtered_df.columns:
        filtered_df['ageOption'] = ""

    weighted_texts = []
    for idx, row in filtered_df.iterrows():
        pseudo_row = {
            'mbti': row.get('mbti', ""),
            'contactFrequencyOption': row.get('contactFrequencyOption', ""),
            'bigHobby': row.get('bigHobby', ""),
            'ageOption': row.get('ageOption', "")
        }
        wtext = build_weighted_text_for_row(
            pseudo_row, mbtiWeight, contactFrequencyWeight, hobbyWeight, ageWeight
        )
        weighted_texts.append(wtext)

    filtered_df['weighted_text'] = weighted_texts
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(filtered_df['weighted_text'])

    return {
        'df': filtered_df,
        'tfidf_matrix': tfidf_matrix,
        'vectorizer': tfidf
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="소분류 취미를 대분류로 변환")
    parser.add_argument("--hobbies", type=str, required=True, help="소분류 취미 입력 (예: '클라이밍, 뮤지컬')")

    args = parser.parse_args()

    # ✅ 입력된 취미 목록 가져오기
    input_hobbies = [h.strip() for h in args.hobbies.split(',') if h.strip()]

    # ✅ 소분류 → 대분류 변환
    converted_hobbies = {hobby: map_small_to_big(hobby, BIG_CATEGORY_DICT) for hobby in input_hobbies}

    # ✅ 결과 출력
    print("\n✅ 입력된 소분류 취미:", input_hobbies)
    print("✅ 변환된 대분류 취미:")
    for small, big in converted_hobbies.items():
        if big:
            print(f"   - {small} → {big}")
        else:
            print(f"   - {small} → (대분류 없음)")

    # ✅ JSON 형태로 변환하여 출력 가능
    import json
    print("\n✅ 변환 결과 (JSON):")
    print(json.dumps(converted_hobbies, indent=4, ensure_ascii=False))