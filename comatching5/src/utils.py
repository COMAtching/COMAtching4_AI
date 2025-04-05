import os
import argparse
import json
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError, AuthenticationError
from prompts import MANAGER_PROMPT
from functions import preprocess_hobby

# ✅ 1. 환경 변수 및 API 키 로드
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY or not API_KEY.startswith("sk-"):
    raise ValueError("❌ 올바른 OpenAI API 키가 설정되지 않았습니다.")

# ✅ 2. 최신 openai 클라이언트 객체 생성
client = OpenAI(api_key=API_KEY)


# ✅ 3. 소분류 취미 추출 함수
def extract_sub_hobbies(user_profile):
    return preprocess_hobby(user_profile)


# ✅ 4. GPT 호출 클래스
class GPTClassifier:
    def __init__(self, client_instance):
        self.client = client_instance

    def classify_hobbies(self, extracted_info, user_profile):
        messages = [
            {"role": "system", "content": MANAGER_PROMPT},
            {"role": "user", "content": (
                f"현재 프로필: {user_profile}\n"
                f"추출된 정보(소분류 취미): {extracted_info}\n"
                f"각 취미를 적절한 대분류 카테고리로 변환하고, JSON 형식으로 반환하세요."
            )}
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0
            )

            result_text = response.choices[0].message.content.strip()

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

        except AuthenticationError:
            raise ValueError("❌ API 인증 실패: API 키가 잘못되었거나 만료되었습니다.")

        except OpenAIError as e:
            print("GPT 호출 오류 발생:", str(e))
            return {"error": str(e)}


# ✅ 5. GPT 응답 파싱 함수
def parse_gpt_response(gpt_response):
    try:
        if isinstance(gpt_response, str) and gpt_response.startswith("{"):
            return json.loads(gpt_response)

        result_dict = {}
        lines = gpt_response.split("\n")
        for line in lines:
            if ":" in line:
                category, hobbies = line.split(":", 1)
                category = category.strip()
                hobbies_list = [h.strip() for h in hobbies.split(",")]
                result_dict[category] = len(hobbies_list)

        return result_dict

    except Exception as e:
        print("❌ GPT 응답 파싱 오류:", str(e))
        return {}


# ✅ 6. CLI 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT를 사용한 취미 변환")
    parser.add_argument("--hobbies", type=str, required=True, help="소분류 취미 입력 (예: '클라이밍, 뮤지컬')")

    args = parser.parse_args()
    user_profile = {"hobbyOption": args.hobbies}
    extracted_hobbies = extract_sub_hobbies(user_profile)

    classifier = GPTClassifier(client)
    result = classifier.classify_hobbies(extracted_hobbies, user_profile)

    print(f'{{"small_category": "{result["small_category"]}"}}')
    print(f'{{"big_category": "{result["big_category"]}"}}')
