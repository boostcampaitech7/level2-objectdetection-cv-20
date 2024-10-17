#!/bin/bash

# 파라미터의 배열을 정의
## 파라미터 전체 조합 실행 (예시는 3 * 3 으로 총 9번 실행)
learning_rates=(0.001 0.0005 0.0001)
batch_sizes=(16 32 64)

# 각 파라미터 조합을 반복
for lr in "${learning_rates[@]}"; do
  for bs in "${batch_sizes[@]}"; do
  # 학습 진행 시 출력(echo = print)
    echo "Running training with learning rate $lr and batch size $bs"

    # Python 스크립트를 nohup으로 실행하고, 결과는 개별 로그 파일에 저장
    nohup python trainer_test.py --learning_rate $lr --batch_size $bs > "./trainer_log/output_lr${lr}_bs${bs}.log" 2>&1 &

  done
done