#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
소분류 → 대분류 매핑 스크립트 (딕셔너리 + GPT backup)
"""
import argparse
import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from utils import GPTClassifier, parse_gpt_response

# ───────────────────── Logger ─────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ───────────────── 딕셔너리 ─────────────────
BIG_CATEGORY_DICT = {
    "스포츠":  ["축구", "농구", "야구", "배드민턴", "헬스", "필라테스",
              "클라이밍", "자전거", "러닝", "운동하기", "테니스치기"],
    "문화":    ["독서", "영화", "웹툰", "OTT 시청", "방탈출", "뮤지컬", "전시회", "페스티벌"],
    "예술":    ["인디음악", "힙합", "발라드", "k-pop", "해외팝송", "악기 연주", "댄스", "패션", "사진 촬영"],
    "여행":    ["국내 여행", "해외 여행", "맛집 탐방", "캠핑", "등산", "드라이브"],
    "자기계발": ["요리", "베이킹", "수집", "뜨개질", "티", "투자", "휴식", "코딩"],
    "게임":    ["보드게임", "롤", "롤토체스", "피파", "배그", "오버워치", "스팀게임", "모바일게임"],
}

# ────────────── 공백·대소문자 무시 매핑 ──────────────
def _norm(txt: str) -> str:
    """공백 제거 + 소문자 → 비교용 키"""
    return "".join(txt.split()).lower()

ALIAS_MAP = {}
for big, subs in BIG_CATEGORY_DICT.items():
    ALIAS_MAP[_norm(big)] = big
    for s in subs:
        ALIAS_MAP[_norm(s)] = big

# ───────────────── Helper ─────────────────
def get_big_category_from_dict(cat: Optional[str]) -> Optional[str]:
    """정규화 키로 바로 조회."""
    if not cat:
        return None
    return ALIAS_MAP.get(_norm(cat))

# 이하 함수들은 그대로 ----------------------------------------------------------
def classify_category(uuid: str, subcat: str, gpt: Optional[GPTClassifier]) -> str:
    big = get_big_category_from_dict(subcat)
    if big:
        return big
    if gpt is None:
        return "기타"
    resp = gpt.classify_hobbies([subcat], {"uuid": uuid, "hobbyOption": subcat})
    big = parse_gpt_response(resp).get("bigcategory", "").strip()
    return big or "기타"

def classify_categories(uuid: str, subcats: List[str], gpt: Optional[GPTClassifier]) -> str:
    big_list = [classify_category(uuid, s.strip(), gpt) for s in subcats if s.strip()]
    return "{" + ", ".join(f'"{b}"' for b in big_list) + "}" if big_list else "None"

def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    parser = argparse.ArgumentParser(description="uuid·소분류 → 대분류 매퍼")
    parser.add_argument("--uuid", default="default_uuid")
    parser.add_argument("--subcategory", nargs="+", default=["축구"])
    args = parser.parse_args()

    subcats = (
        [s.strip() for s in args.subcategory[0].split(",")]
        if len(args.subcategory) == 1 and "," in args.subcategory[0]
        else args.subcategory
    )

    logger.info("▶ START uuid=%s | subcategories=%s", args.uuid, subcats)
    gpt = GPTClassifier(api_key=api_key) if api_key else None
    print("UUID:", args.uuid)
    print("대분류:", classify_categories(args.uuid, subcats, gpt))

if __name__ == "__main__":
    main()