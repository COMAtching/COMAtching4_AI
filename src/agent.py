import os
import sys
from dotenv import load_dotenv
import openai

# .env 파일 로드
load_dotenv()

# ✅ 현재 스크립트의 경로를 기준으로 sys.path에 추가하여 import 문제 해결
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompts import *

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

def manager_agent(extracted_info, user_profile, extractor_message):
    """
    GPT로부터 소분류 취미 -> 대분류 취미 분류 결과를 받음.
    """
    current_profile = {k: v if v else "정보 없음" for k, v in user_profile.items()}
    messages = [
        {"role": "system", "content": MANAGER_PROMPT},
        {
            "role": "user",
            "content": (
                f"현재 프로필: {current_profile}\n"
                f"추출된 정보(소분류 취미): {extracted_info}\n"
                f"추출 모델 메시지: {extractor_message}"
            )
        }
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0
        )
        evaluation = response["choices"][0]["message"]["content"].strip()

        # ✅ 기존 문자열 반환 대신 JSON 형태로 변환하여 반환
        return {
            "subcategory": extracted_info,
            "bigcategory": evaluation
        }

    except Exception as e:
        print("오류 발생:", e)
        return {"error": str(e)}