# import os
# import openai
# import pandas as pd
# import argparse  # ✅ CLI 입력을 받을 수 있도록 argparse 추가
# import json  # ✅ 변환된 데이터를 JSON으로 변환하기 위해 추가
# from dotenv import load_dotenv
# from prompts import MANAGER_PROMPT
# from functions import preprocess_hobby
#
# # 1️. 환경 변수 및 .env 파일에서 API 키 로드
# load_dotenv()  # .env 파일에서 환경 변수 로드
# API_KEY = os.getenv("OPENAI_API_KEY")
#
# if not API_KEY or not API_KEY.startswith("sk-"):
#     raise ValueError("❌ 올바른 OpenAI API 키가 설정되지 않았습니다. .env 파일 또는 환경 변수를 확인하세요.")
#
#
# # 2. 소분류 취미 추출 함수
# def extract_sub_hobbies(user_profile):
#     """
#     유저 프로필에서 소분류 취미를 추출하는 함수
#     """
#     return preprocess_hobby(user_profile)
#
#
# # 3. GPT 호출 클래스 (소분류 -> 대분류 변환)
# class GPTClassifier:
#     def __init__(self, api_key=API_KEY):
#         """
#         GPTClassifier 초기화 (API 키 설정)
#         """
#         self.api_key = api_key
#         openai.api_key = self.api_key
#
#     def classify_hobbies(self, extracted_info, user_profile, extractor_message="추가 분류 작업을 진행해주세요."):
#         """
#         GPT를 이용하여 소분류 취미를 대분류로 변환
#         """
#         messages = [
#             {"role": "system", "content": MANAGER_PROMPT},
#             {"role": "user", "content": (
#                 f"현재 프로필: {user_profile}\n"
#                 f"추출된 정보(소분류 취미): {extracted_info}\n"
#                 f"추출 모델 메시지: {extractor_message}"
#             )}
#         ]
#
#         try:
#             response = openai.ChatCompletion.create(
#                 model="gpt-4o",
#                 messages=messages,
#                 temperature=0
#             )
#             result_text = response["choices"][0]["message"]["content"].strip()
#
#             # ✅ JSON 변환 추가 (소분류 + 대분류 함께 반환)
#             result_json = {
#                 "subcategory": extracted_info,
#                 "bigcategory": result_text
#             }
#
#             # ✅ 변환된 데이터 터미널에 출력
#             print("\n✅ GPT 응답 데이터:")
#             print(json.dumps(result_json, indent=4, ensure_ascii=False))
#
#             return result_json
#
#         except openai.error.AuthenticationError:
#             raise ValueError("❌ API 인증 실패: API 키가 잘못되었거나 만료되었습니다. 올바른 키를 입력하세요.")
#
#         except openai.error.OpenAIError as e:
#             print("GPT 호출 오류 발생:", str(e))
#             return {"error": str(e)}
#
#
# ## 4. GPT 응답 파싱 (소분류 → 대분류 매핑)
# # def parse_gpt_response(gpt_response):
# #     """
# #     GPT 응답을 파싱하여 {대분류: 등장횟수} 형태로 변환 → 단순한 'bigcategory'로 반환하도록 수정
# #     """
# #     # ✅ 기존에 있던 횟수 파싱 제거 (그냥 대분류 값만 반환)
# #     return gpt_response.strip()
# def parse_gpt_response(gpt_response):
#     """
#     GPT 응답을 파싱하여 {대분류: 등장 횟수} 형태로 변환
#     """
#     try:
#         # ✅ 만약 JSON 형식이면 파싱
#         if isinstance(gpt_response, str) and gpt_response.startswith("{"):
#             return json.loads(gpt_response)
#
#         # ✅ 텍스트를 수동으로 파싱 (예: "예술: 음악감상, 사진")
#         result_dict = {}
#         lines = gpt_response.split("\n")
#         for line in lines:
#             if ":" in line:
#                 category, hobbies = line.split(":", 1)
#                 category = category.strip()
#                 hobbies_list = [h.strip() for h in hobbies.split(",")]
#
#                 # 등장 횟수를 1로 설정
#                 result_dict[category] = len(hobbies_list)
#
#         return result_dict
#
#     except Exception as e:
#         print("❌ GPT 응답 파싱 오류:", str(e))
#         return {}
#
#
# # ✅ CLI에서 실행 가능하도록 argparse 추가
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="GPT를 사용한 취미 변환")
#     parser.add_argument("--hobbies", type=str, required=True, help="소분류 취미 입력 (예: '클라이밍, 뮤지컬')")
#
#     args = parser.parse_args()
#
#     user_profile = {"hobbyOption": args.hobbies}
#     classifier = GPTClassifier()
#     extracted_hobbies = extract_sub_hobbies(user_profile)
#     result = classifier.classify_hobbies(extracted_hobbies, user_profile)
#
#     print("\n✅ 최종 변환된 데이터:")
#     print(json.dumps(result, indent=4, ensure_ascii=False))
#
#     print("\n✅ 변환된 대분류 취미:")
#     print(result["bigcategory"])

import os
import openai
import argparse  # CLI 입력을 받을 수 있도록 argparse 추가
import json  # 변환된 데이터를 JSON으로 변환하기 위해 추가
from dotenv import load_dotenv
from prompts import MANAGER_PROMPT
from functions import preprocess_hobby

# 1️. 환경 변수 및 .env 파일에서 API 키 로드
load_dotenv()  # .env 파일에서 환경 변수 로드
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY or not API_KEY.startswith("sk-"):
    raise ValueError("❌ 올바른 OpenAI API 키가 설정되지 않았습니다. .env 파일 또는 환경 변수를 확인하세요.")

# OpenAI API 키 설정 (최신 버전 호환)
openai.api_key = API_KEY


# 2. 소분류 취미 추출 함수
def extract_sub_hobbies(user_profile):
    """
    유저 프로필에서 소분류 취미를 추출하는 함수
    """
    return preprocess_hobby(user_profile)


# 3. GPT 호출 클래스 (소분류 -> 대분류 변환)
class GPTClassifier:
    def __init__(self, api_key=None):
        """
        GPTClassifier 초기화 (API 키 설정)
        """
        if api_key:
            openai.api_key = api_key  # ✅ API 키 설정

    def classify_hobbies(self, extracted_info, user_profile):
        """
        GPT를 이용하여 소분류 취미를 대분류로 변환
        """
        messages = [
            {"role": "system", "content": MANAGER_PROMPT},
            {"role": "user", "content": (
                f"현재 프로필: {user_profile}\n"
                f"추출된 정보(소분류 취미): {extracted_info}\n"
                f"각 취미를 적절한 대분류 카테고리로 변환하고, JSON 형식으로 반환하세요."
            )}
        ]

        try:
            response = openai.ChatCompletion.create(  # ✅ 최신 OpenAI API 적용
                model="gpt-4o",
                messages=messages,
                temperature=0
            )
            result_text = response["choices"][0]["message"]["content"].strip()

            # ✅ JSON 형식으로 변환
            category_mapping = {}
            lines = result_text.split("\n")
            for line in lines:
                if ":" in line:
                    small, big = line.split(":")
                    category_mapping[small.strip()] = big.strip()

            small_category_list = list(category_mapping.keys())
            big_category_list = list(set(category_mapping.values()))  # 중복 제거

            result_json = {
                "small_category": ", ".join(small_category_list),
                "big_category": " ".join(big_category_list)
            }

            return result_json

        except openai.error.AuthenticationError:
            raise ValueError("❌ API 인증 실패: API 키가 잘못되었거나 만료되었습니다. 올바른 키를 입력하세요.")

        except openai.error.OpenAIError as e:
            print("GPT 호출 오류 발생:", str(e))
            return {"error": str(e)}

# ✅ 4. GPT 응답을 가공하는 `parse_gpt_response` 함수 (GPTClassifier 아래에 추가)
def parse_gpt_response(gpt_response):
    """
    GPT 응답을 파싱하여 대분류 취미 목록을 반환하는 함수
    """
    try:
        # JSON 형식이면 직접 변환
        if isinstance(gpt_response, str) and gpt_response.startswith("{"):
            return json.loads(gpt_response)

        # 텍스트 파싱 (예: "스포츠: 축구, 농구\n문화: 독서, 영화")
        result_dict = {}
        lines = gpt_response.split("\n")
        for line in lines:
            if ":" in line:
                category, hobbies = line.split(":", 1)
                category = category.strip()
                hobbies_list = [h.strip() for h in hobbies.split(",")]

                # 등장 횟수를 1로 설정
                result_dict[category] = len(hobbies_list)

        return result_dict

    except Exception as e:
        print("❌ GPT 응답 파싱 오류:", str(e))
        return {}

# CLI에서 실행 가능하도록 argparse 추가
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT를 사용한 취미 변환")
    parser.add_argument("--hobbies", type=str, required=True, help="소분류 취미 입력 (예: '클라이밍, 뮤지컬')")

    args = parser.parse_args()

    user_profile = {"hobbyOption": args.hobbies}
    classifier = GPTClassifier()
    extracted_hobbies = extract_sub_hobbies(user_profile)
    result = classifier.classify_hobbies(extracted_hobbies, user_profile)

    # 최종 결과 출력 형태 조정
    print(f'{{"small_category": {result["small_category"]}}}')
    print(f'{{"big_category": {result["big_category"]}}}')