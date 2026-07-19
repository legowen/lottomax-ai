# LottoMax AI — Multi-Stage Completion Plan

이 문서는 미뤄뒀던 프로젝트를 다단계 서브에이전트 파이프라인으로 완주하기 위한 실행 계획이다.

## Stage 0 — Orchestration & Environment
- 실행 계획 고정 (이 문서)
- 백엔드 venv 구성, 의존성 설치 (TensorFlow 포함)

## Stage 1 — Research (리서치 서브에이전트)
- `data/LOTTOMAX.csv` 전수 분석: 회차 수, 기간, **7/49 → 7/50 게임 규칙 변경 시점** 탐지
- 균등성 검정(chi-square), 자기상관, 페어 분포 vs 이론치
- 기존 6개 전략의 예측력 백테스트 (랜덤 베이스라인 대비, 기대 매치수 = 7×7/50 ≈ 0.98)
- "당첨 가능성을 실질적으로 개선할 수 있는" 수단 조사:
  - 번호 적중률 자체는 개선 불가(추첨은 무작위) — 정직하게 검증
  - **기대값(EV) 최적화**: 인기 조합 회피로 당첨 시 분배금 극대화 (파리뮤추얼 구조)
- 산출물: `docs/RESEARCH.md`

## Stage 2 — Implementation (구현 서브에이전트)
- 리서치 결과 기반 수정:
  - 7/49→7/50 시대(era) 인식 통계 (빈도/갭 왜곡 제거)
  - TensorFlow 지연 임포트 (TF 없이도 서버 구동)
  - **Strategy 7: Smart Pick (EV 최적화)** — 인기 조합 회피 필터
  - `/backtest` 엔드포인트 + 백테스트 엔진 (전략별 성능 vs 랜덤)
  - pytest 테스트 스위트
  - 프론트엔드: 백테스트 탭, EV 전략 노출, 정직한 안내 문구, 이미지 자리표시자(추후 사용자가 이미지 추가)

## Stage 3 — Adversarial Verification (적대적 검증 서브에이전트)
- 구현을 공격: 통계 주장 검증, 엣지케이스 폭격, 서버 기동/엔드포인트 퍼징, 테스트/빌드 실행
- 발견된 결함은 심각도별 보고 → 수정 루프

## Stage 4 — End-to-End Run
- 백엔드 기동 → 학습(단축 epochs) → 예측 → 백테스트 → 프론트 빌드 검증
- 커밋 & 푸시 (`claude/multi-stage-subagent-project-yvum42`)
