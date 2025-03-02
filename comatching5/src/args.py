# src/args.py

import argparse


def get_args():
    """
    명령줄 인자로 --mbti_weight, --contact_weight, --hobby_weight 세 가지를 받아오는 예시.
    기본값은 모두 1.0
    """
    parser = argparse.ArgumentParser(description="Cosine Similarity Weights")
    parser.add_argument('--m', type=float, default=1.0, help='mbti_weight')
    parser.add_argument('--c', type=float, default=1.0, help='contact_weight')
    parser.add_argument('--h', type=float, default=1.0, help='hobby weight')
    parser.add_argument('--subcategory', type=str, required=True, help="소분류 데이터 (예: '축구, 피파, 피트니스')")


    args = parser.parse_args()
    return args
