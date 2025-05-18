#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPT 보조 유틸 · GPTClassifier
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError, AuthenticationError

from prompts import MANAGER_PROMPT
from functions import preprocess_hobby


# ────────────────────────── 공통 유틸 ──────────────────────────
def extract_sub_hobbies(user_profile: Dict[str, str]) -> List[str]:
    """프로필 dict에서 소분류 취미 리스트 추출."""
    return preprocess_hobby(user_profile)


def parse_gpt_response(gpt_response) -> Dict[str, str]:
    """
    GPT 응답(dict 또는 JSON str)을 { "bigcategory": "<대분류>" } 로 정규화.
    """
    if isinstance(gpt_response, dict):
        return gpt_response
    try:
        return json.loads(gpt_response)
    except Exception as e:
        print("❌ GPT 응답 파싱 오류:", e)
        return {}


# ────────────────────────── GPT 분류기 ──────────────────────────
class GPTClassifier:
    """
    소분류 취미 → 대분류 카테고리 매핑을 GPT-4o 모델로 수행.
    반환 형식: {"bigcategory": "<스포츠 ...>"}  (여러 개면 공백 구분)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key or not self.api_key.startswith("sk-"):
            raise ValueError("❌ OPENAI_API_KEY가 설정되지 않았습니다.")

        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature

    # ------------------------------------------------------------------ #
    def classify_hobbies(
        self,
        extracted_info: List[str],
        user_profile: Dict[str, str],
    ) -> Dict[str, str]:
        """소분류 목록을 GPT에 보내 대분류 하나(or 여러 개)로 매핑."""
        messages = [
            {"role": "system", "content": MANAGER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"현재 프로필: {user_profile}\n"
                    f"추출된 소분류 취미: {extracted_info}\n\n"
                    "각 항목을 다음 후보 중 하나로 매핑하십시오:\n"
                    "스포츠, 문화, 예술, 여행, 자기계발, 게임, 기타\n"
                    '반드시 JSON 형식 `{"bigcategory": "<대분류들(공백 구분)>"}`로만 답하세요.'
                ),
            },
        ]

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                # 🎯 JSON 강제
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)  # {"bigcategory": "..."}
        except AuthenticationError:
            raise ValueError("❌ API 인증 실패: 키가 잘못되었거나 만료되었습니다.")
        except OpenAIError as e:
            print("GPT 호출 오류:", e)
            return {"bigcategory": "기타"}


# ────────────────────────── CLI DEMO ──────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT 취미 대분류 변환 DEMO")
    parser.add_argument("--hobbies", required=True, help="예: '클라이밍, 뮤지컬'")
    args = parser.parse_args()

    profile = {"hobbyOption": args.hobbies}
    classifier = GPTClassifier()
    extracted = extract_sub_hobbies(profile)
    result = classifier.classify_hobbies(extracted, profile)
    print(json.dumps(result, ensure_ascii=False, indent=2))