#!/bin/bash

# 파라미터의 배열을 정의
## 파라미터 전체 조합 실행 (예시는 3 * 3 으로 총 9번 실행)
# score=(0.1 0.5 0.7)
# pos=(0.9 0.7 0.5)
# neg=(0.5 0.3 0)
# min=(0.3 0.4 0.5)
score=(0.1 0.5)
pos=(0.9 0.1)

# 각 파라미터 조합을 반복
for po in "${pos[@]}"; do
  for sc in "${score[@]}"; do
  # 학습 진행 시 출력(echo = print)
    echo "Running training with score $sc, pos $po"

    # Python 스크립트를 nohup으로 실행하고, 결과는 개별 로그 파일에 저장
    nohup python trainer_test.py --score $sc --pos $po > "./trainer_log/output_sc${sc}_po${po}.log" 2>&1 &
  done
done

wait
