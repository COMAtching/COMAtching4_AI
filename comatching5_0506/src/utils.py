import os
import json
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError, AuthenticationError
from prompts import MANAGER_PROMPT
from functions import preprocess_hobby

# 환경 변수 로드
load_dotenv()

# ✅ GPT 분류기 클래스
class GPTClassifier:
    def __init__(self, api_key):
        if not api_key or not api_key.startswith("sk-"):
            raise ValueError("❌ 올바른 OpenAI API 키가 설정되지 않았습니다.")
        self.client = OpenAI(api_key=api_key)

    def classify_hobbies(self, extracted_info, user_profile):
        messages = [
            {"role": "system", "content": MANAGER_PROMPT},
            {"role": "user", "content": (
                f"현재 프로필: {user_profile}\n"
                f"추출된 정보(소분류 취미): {extracted_info}\n"
                f"각 취미를 다음 대분류 카테고리(스포츠, 문화, 예술, 여행, 자기계발, 게임) 중 하나로 분류하고, JSON 형식으로 반환하세요."
            )}
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0
            )

            result_text = response.choices[0].message.content.strip()
            print("GPT 응답 내용:\n", result_text)  # 디버깅용 출력

            # 마크다운 코드 블록 제거
            if result_text.startswith("```json") and result_text.endswith("```"):
                result_text = result_text[7:-3].strip()
            elif result_text.startswith("```") and result_text.endswith("```"):
                result_text = result_text[3:-3].strip()

            # JSON 파싱 시도
            try:
                category_mapping = json.loads(result_text)
            except json.JSONDecodeError:
                # Fallback 파싱
                category_mapping = {}
                for line in result_text.split("\n"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        category_mapping[k.strip()] = v.strip().strip('"')

            # 대분류 수집, 따옴표 제거
            big_categories = [val.strip('"') for val in category_mapping.values()]
            big_categories = list(set(big_categories))  # 중복 제거
            return {"bigcategory": big_categories[0] if big_categories else ""}

        except AuthenticationError:
            raise ValueError("❌ API 인증 실패: API 키가 잘못되었거나 만료되었습니다.")

        except OpenAIError as e:
            print("GPT 호출 오류 발생:", str(e))
            return {"error": str(e)}


# ✅ 보조 함수: 소분류 추출 (현재는 사용 안 함이지만 남겨둠)
def extract_sub_hobbies(user_profile):
    return preprocess_hobby(user_profile)


# ✅ GPT 응답 파싱 함수 (구 버전과 호환성 있음)
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