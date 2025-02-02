# agent.py
import openai
from .prompts import *

# GPT-4o 모델 사용 시 아래와 같이 설정
openai.api_key = 'OPENAI_API_KEY'

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
        evaluation = response.choices[0].message.content.strip()

        # 기존: print(f"\n[GPT 응답]:\n{evaluation}")
        # ----> 이 부분을 제거하거나 주석 처리
        # print(f"\n[GPT 응답]:\n{evaluation}")

        return evaluation

    except Exception as e:
        print("오류 발생:", e)
        return ""