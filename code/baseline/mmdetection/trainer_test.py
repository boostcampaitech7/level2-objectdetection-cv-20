import argparse
# 명령줄 인자 파서 설정
parser = argparse.ArgumentParser(description="Model Training Script")
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
args = parser.parse_args()

# 파라미터로 받은 값 출력 (학습에 사용할 수 있음)
print(f"Learning rate: {args.learning_rate}")
print(f"Batch size: {args.batch_size}")

# 여기서 args.learning_rate와 args.batch_size를 모델 학습에 사용할 수 있음
# 예시로는 학습 코드를 넣어두면 돼
# model.train(learning_rate=args.learning_rate, batch_size=args.batch_size)