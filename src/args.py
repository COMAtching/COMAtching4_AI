import argparse


def get_args():
    """
    명령줄 인자로 --uuid, --subcategory, --mbti_weight, --contact_weight, --hobby_weight를 받아옵니다.
    uuid와 subcategory는 BE에서 전달되는 값이므로 기본값을 제공하지 않습니다.
    """
    parser = argparse.ArgumentParser(description="UUID, 소분류 및 가중치 인자를 입력받습니다.")
    parser.add_argument('--uuid', type=str, help='사용자 UUID (BE에서 전달됨)')
    parser.add_argument('--subcategory', type=str, help="소분류 데이터 (예: '축구, 피파, 피트니스') (BE에서 전달됨)")
    parser.add_argument('--m', type=float, default=1.0, help='mbti_weight (기본: 1.0)')
    parser.add_argument('--c', type=float, default=1.0, help='contact_weight (기본: 1.0)')
    parser.add_argument('--h', type=float, default=1.0, help='hobby_weight (기본: 1.0)')

    args = parser.parse_args()
    return args
