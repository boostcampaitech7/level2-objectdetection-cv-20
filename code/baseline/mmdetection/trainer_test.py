import argparse
import os
# 명령줄 인자 파서 설정
parser = argparse.ArgumentParser(description="Model Training Script")
parser.add_argument('--score', type=float, default=0.1, help='Score threshold')
parser.add_argument('--pos', type=float, default=0.9, help='Positive threshold')
parser.add_argument('--neg', type=float, default=0.5, help='Negative threshold')
parser.add_argument('--min', type=float, default=0.3, help='Minimum threshold')
args = parser.parse_args()

# 파라미터로 받은 값 출력 (학습에 사용할 수 있음)
print(f"Score threshold: {args.score}")
print(f"Positive threshold: {args.pos}")
print(f"Negative threshold: {args.neg}")
print(f"Minimum threshold: {args.min}")

# 여기서 args.score, args.pos, args.neg, args.min을 모델 학습에 사용할 수 있음
# 예시로는 학습 코드를 넣어두면 돼
# model.train(score=args.score, pos=args.pos, neg=args.neg, min=args.min)

print("Training completed.")
os._exit(0)