# 신용카드 고객 세그먼트 분류 프로젝트 (v3.5 기준)

> **Baseline**: v3.5 (Hybrid Top150 + 도메인 파생변수 6개 + XGBoost, depth=6)  
> **Target**: 5개 Segment(0~4) 안정 분류 + 희귀 세그먼트(Segment 0,1) 행동 이해 및 전략 활용

---

## 0. 프로젝트 목적

이 프로젝트의 목적은 두 가지다.

1. **신용카드 고객을 5개 Segment(0~4)로 안정적으로 분류하는 머신러닝 모델 구축**
2. **각 세그먼트, 특히 희귀 세그먼트(Segment 0, 1)의 행동 패턴을 이해하고,  
   향후 마케팅/리스크 전략으로 확장 가능한 인사이트 확보**

현재까지는 **v3.5 모델**을 공식 베이스라인으로 둔다.

- 입력: Hybrid Top150 피처 + 도메인 파생변수 6개 (총 156개)
- 모델: XGBoost (multi:softprob, depth=6)
- 성능: Validation Macro F1 ≈ **0.5313**

이 README는 **여기까지의 EDA + 모델링 진행 내역 + 시행착오 + v4 방향**을 한 번에 정리한 문서다.

---

## 1. 데이터 개요 및 세그먼트 분포

### 1.1 데이터 구조

- 파일: `df_master_preprocessed_v1.parquet`
- 크기: **(400,000, 851)**
  - 400,000명 고객
  - 851개 피처  
    (고객 기본 특성, 이용금액/건수, 청구·입금, 한도·잔액, 포인트/마일리지, 날짜/경과일 등)
- 타깃 변수: `Segment` (0, 1, 2, 3, 4) — 다중 클래스 분류 문제

### 1.2 Train / Validation 분할 (v2 ~ v3.5 공통)

```python
X_all = df_master.drop(columns=["Segment"])
y_all = df_master["Segment"]

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all,
)
Train: 320,000건 (80%)

Val: 80,000건 (20%)

stratify=y로 세그먼트 분포 보존

Train 세그먼트 분포

Segment 4: 256,274

Segment 3: 46,565

Segment 2: 17,012

Segment 0: 130

Segment 1: 19

Validation 세그먼트 분포

Segment 4: 64,068

Segment 3: 11,642

Segment 2: 4,253

Segment 0: 32

Segment 1: 5

🔎 핵심

Segment 0, 1: 데이터 수가 극단적으로 적은 희귀 클래스

Segment 4: 전체의 대부분을 차지하는 초다수 클래스

→ 전형적인 심각한 다중 클래스 불균형 문제

2. Global Feature Importance (Hybrid Top150)
2.1 Hybrid Top150 선정 방법
전체 피처(851개)에 대해:

XGBoost 중요도 (gain) 계산

Mutual Information (MI) 계산

각 랭킹에서 피처별 순위 산출 후,

final_rank
=
xgb_rank
+
mi_rank
2
final_rank= 
2
xgb_rank+mi_rank
​
 
final_rank 기준 상위 150개 피처를 **top150_final**로 저장

파일: features/top150_final.parquet

→ 이 150개에 도메인 기반 파생변수 6개(v3_FE) 를 추가한 것이 v3.5의 최종 입력 피처 세트(156개)

2.2 상위 핵심 피처 예시
Hybrid Top150 상단에는 다음 피처들이 위치한다.

이용금액대

정상청구원금_B5M

이용금액_오프라인_R3M

이용금액_R3M_신용체크

이용금액_오프라인_B0M

정상청구원금_B2M

정상청구원금_B0M

이용건수_신용_R12M

이용금액_일시불_R12M

최대이용금액_일시불_R12M

연속유실적개월수_기본_24M_카드

…

2.3 중요도 결과가 말해주는 것
① 이용금액 / 청구금액 축

이용금액대, 이용금액_*, 정상청구원금_*
→ “얼마나, 어떤 패턴으로 쓰는지”가 세그먼트 구분의 1차 기준.

② 기간/윈도우 축 (R3/R6/R12, B5/B2/B0)

R3 / R6 / R12, B5 / B2 / B0 등 다양한 기간 창으로 나뉜 피처들이 상위에 많이 등장
→ 단기 vs 장기 이용 추세 차이로 세그먼트가 갈린다.

③ 활동 빈도 / 지속성 축

이용건수_신용_R12M, 연속유실적개월수_기본_24M_카드
→ 장기간 꾸준히 사용하는 고객 vs 특정 기간에만 사용하는 고객을 구분.

3. 세그먼트별 평균 패턴 (예시)
대표 피처 몇 개에 대한 Segment별 평균값은 대략 다음과 같다.
(실제 숫자는 EDA Notebook 기준)

Feature	Seg0	Seg1	Seg2	Seg3	Seg4
이용금액대	0.00	0.00	0.58	1.12	3.13
정상청구원금_B5M	50,846	50,260	21,919	11,847	3,616
이용금액_오프라인_R3M	59,214	52,731	31,292	23,112	6,769
이용금액_R3M_신용체크	98,271	90,748	60,668	38,360	9,771
이용금액_오프라인_B0M	20,567	17,118	10,427	7,612	2,192
정상청구원금_B2M	39,009	37,867	18,565	10,286	3,176
정상청구원금_B0M	37,679	37,144	18,476	10,055	3,052
이용건수_신용_R12M	637.7	659.0	470.6	366.8	123.6
이용금액_일시불_R12M	389,026	325,289	179,203	99,033	26,636
최대이용금액_일시불_R12M	68,939	47,605	32,522	17,834	5,674

3.1 해석 요약
Segment 4

이용금액/청구금액/건수가 전반적으로 가장 낮음
→ “저활동·소액 위주의 대규모 일반 고객층”

Segment 3, 2

4 → 3 → 2로 갈수록 이용/청구 규모가 증가
→ “중간/상위 활동 고객층”

Segment 0, 1 (희귀)

여러 지표에서 2,3보다도 더 높은 값

→ **“과거 고액/고빈도 이용 이력이 있는 특수 고객군”**으로 해석 가능

4. Rare Segment(0,1) vs Others(2,3,4) 분석
4.1 Rare Difference Score 정의
features/segment_stats_summary.csv는 아래와 같이 생성했다.

python
코드 복사
seg_stats = df.groupby("Segment").agg(["mean", "std", "median"])
seg_stats.to_csv("segment_stats_summary.csv", encoding="utf-8-sig")
각 feature에 대해:

mean_diff = |mean0 - mean_others| + |mean1 - mean_others|

median_diff, std_diff도 같은 방식

최종 스코어:

rare_difference_score
=
0.5
⋅
mean_diff
+
0.3
⋅
median_diff
+
0.2
⋅
std_diff
rare_difference_score=0.5⋅mean_diff+0.3⋅median_diff+0.2⋅std_diff
→ 이 값이 클수록 Segment 0/1과 2/3/4의 행동 차이가 큰 피처

상위 150개는 features/rare_top150_from_seg_stats.csv로 저장.

4.2 rare_difference_score 상위 피처 패턴
(1) 날짜 / 최근성(Recency) 지표 – 최상위
rv최초시작후경과일

최종이용일자_CA

최종이용일자_할부

최종이용일자_일시불

최종이용일자_신판

최종이용일자_기본

최종이용일자_카드론

최종카드발급일자

입회일자_신용

해석

가입 시점, 카드 발급 시점, 마지막 이용 시점, 카드론 이용 종료 시점 등 전체 타임라인이
0/1과 2/3/4 사이에서 크게 다르다.

특히 최종이용일자_* 계열 mean/median 차이가 매우 커서
→ **“과거에는 쓰다가 지금은 거의 안 쓰는, 장기 비활성/패턴 전환 고객”**일 가능성이 크다.

(2) 이용금액 · 청구금액 · 한도 · 평잔 – 돈 흐름 구조 차이
대표 피처:

이용금액_일시불_R12M, 이용금액_일시불_R6M, 이용금액_일시불_R3M

청구금액_R6M, 청구금액_R3M

이용금액_CA_R12M, 이용금액_오프라인_R6M

이용금액_R3M_신용, 이용금액_R3M_신용체크

카드이용한도금액, 카드이용한도금액_B1M, 카드이용한도금액_B2M

평잔_6M, 평잔_3M, 월중평잔 등

해석

Segment 2/3/4:

R3/R6/R12 구간에서 안정적이고 반복적인 이용/청구 패턴

Segment 0/1:

어떤 구간은 거의 0, 어떤 구간은 튀는 값

→ 특정 시점에만 크게 쓰고, 나머지 구간은 비활동

한도·평잔 대비 실제 사용이 비정상적으로 낮은 경우도 존재
→ “슬리핑 하이리밋” 또는 “잔고만 유지하는 비활성 고객” 패턴

(3) 연체 / 카드론 / 현금서비스 / 선입금 – 리스크 · 정리 패턴
연체일수_B1M, 연체일수_B2M, 연체일수_최근

카드론이용금액_누적, 최종카드론_대출금액

잔액_현금서비스_B0M/B1M/B2M

잔액_카드론_B0M~B5M

연체입금원금_*M, 선입금원금_*M 등

해석

일부 rare 고객은

카드론·현금서비스를 한 번 크게 쓰고,

이후 상환·정리 후 비활성화된 패턴을 보일 수 있음.

“연체 → 상환 → 사용 중단” 같은 위험 후 정리 시나리오가 데이터에 녹아 있을 가능성.

(4) 포인트 / 마일리지 / 라이프스타일 지표
포인트/마일리지: 포인트_*, 마일_*

생활/소비: 쇼핑_*, 교통_주유이용금액, 납부_통신비이용금액 등

해석

Segment 2/3/4:

포인트/마일리지 적립·사용, 온라인/오프라인 쇼핑, 교통/통신 납부 등
→ 카드를 생활 전반에 붙여 쓰는 고객층

Segment 0/1:

이 활동들이 거의 없거나, 구조적으로 다름
→ 카드 중심 라이프스타일이 아닌 한시적 사용/비활성 고객층

5. 모델링 타임라인 (v2 → v3 → v3.5)
5.1 v2 – Class-Weighted Baseline
입력 피처: 전체 850개

모델: XGBoost (multi:softprob)

주요 하이퍼파라미터:

max_depth=4

n_estimators=500

learning_rate=0.05

Class weight: class_weight="balanced" → sample_weight로 반영

성능 (Validation)

Macro F1 ≈ 0.521

Segment 2/3/4: F1 약 0.65 ~ 0.93

Segment 0/1: 데이터 희귀로 인해 F1 ≈ 0에 가까움

교훈

전체 피처를 그대로 써도 baseline은 괜찮은 수준

하지만

해석력이 부족하고,

희귀 세그먼트 개선 여지가 큼
→ Feature Selection + Feature Engineering 필요하다고 판단

5.2 v3 – Hybrid Top50 + FE6
Feature Selection:

Hybrid Top50 (XGB + MI)만 사용

Feature Engineering:

도메인 기반 파생변수 6개 (v3_FE6)

최종 피처 수: 56개

depth=4, 6 두 버전 실험

성능

depth=4: Macro F1 ≈ 0.49

depth=6: Macro F1 ≈ 0.516

시행착오 / 교훈

Top20~30처럼 너무 aggressively 피처를 줄이면 성능이 크게 하락

→ 개별 피처 영향은 작아 보여도, 다수 피처 조합 효과가 크다는 것을 확인

Tree depth를 6으로 올리면 비선형 경계 학습이 잘 되어 v2 수준까지 성능 회복 가능

단, Segment 1 문제는 여전히 그대로
→ 데이터 희귀성 자체가 병목이라는 점이 명확해짐

5.3 v3.5 – Top150 + FE6 + depth=6 (현재 베이스라인)
입력 피처:

Hybrid Top150 원본 피처

v3_FE6 (파생변수 6개)

→ 최종 피처 수: 156개

모델:

XGBoost (multi:softprob)

max_depth=6

n_estimators=500

Class weight(balanced) 동일 유지

성능 (Validation)

Macro F1 ≈ 0.5313 (현재까지 최고)

Segment 2/3/4: v2보다 F1/Recall 소폭 개선

Segment 0: 일정 수준 분류 가능

Segment 1: 여전히 예측 어려움 (데이터 19/5의 한계)

정리

v3.5는

v2보다 성능은 소폭 개선

피처 구조는 해석 가능성 대폭 상승

→ v4에서 희귀 세그먼트(0,1) 개선을 위한 확장 베이스라인으로 적합

6. v4를 위한 준비 사항 (이미 정리된 아티팩트)
6.1 Feature 리스트 / FE 후보 파일
features/v3_5_feature_list.csv, features/v3_5_feature_list.txt

v3.5에서 실제 사용한 156개 피처 목록

source 컬럼:

HYBRID_TOP150

CUSTOM_FE_v3

features/rare_top150_from_seg_stats.csv

rare_difference_score 기반으로 뽑은 희귀 세그먼트 민감 피처 Top150

features/v4_FE_candidate_list.json

희귀 세그먼트(0,1)를 강화하기 위해 설계한 추가 파생변수 후보 리스트

각 항목별 정보:

feature_name

base_columns

formula

description_kr (한글 설명)

target_segments (주로 "0", "1", "rare_vs_others")

예시:

v4_last_use_gap_CA
→ today - 최종이용일자_CA (체크/CA 영역 마지막 사용 이후 경과일)

v4_limit_to_usage_ratio_R12M
→ 한도 대비 12M 사용 비율

v4_recent_zero_usage_flag
→ 최근 3개월 완전 미사용 여부

v4_long_inactive_high_limit_flag
→ 한도는 높은데 장기 미사용인 “슬리핑 하이리밋” 플래그

7. v4 이후 모델링 팀이 지향할 방향
7.1 희귀 세그먼트(0,1) 처리 전략
현재까지의 분석/실험을 기반으로, v4/v5에서는 아래 방향을 지향한다.

Oversampling (ADASYN / SMOTE)

Segment 1 (Train 19, Val 5)는 단독 학습이 사실상 불가능

Synthetic sample 생성으로 최소 학습 가능한 수준까지 데이터 보강

Rare-Group 통합 + 2-Stage 모델

1단계: (Segment 0 + 1) vs (2/3/4) 이진 분류

2단계: Rare-group 내부에서 0/1 세분화 시도

→ 금융 도메인에서 희귀 고객군을 다룰 때 자주 쓰는 방식

Recency / Limit / Balance 축 파생변수 적극 활용

v4_FE_candidate_list.json에 정의된 시간·한도·평잔·리스크·라이프스타일 관련 FE 적용

희귀 세그먼트가 “언제까지 쓰다가 끊었는지, 한도 대비 얼마나 안 쓰는지”를 명시적으로 반영

모델 다양화 (선택)

XGBoost 외에:

LightGBM

Focal Loss 적용 모델
등 희귀 클래스에 민감한 알고리즘 실험 가능

8. 리포지토리 구조 & 재현 방법 (v3.5 기준)
8.1 디렉터리 구조 (예시)
text
코드 복사
credit-segmentation-v4/
├─ README.md
├─ .gitignore
├─ features/
│  ├─ top150_final.parquet
│  ├─ v3_5_feature_list.csv
│  ├─ v3_5_feature_list.txt
│  ├─ segment_stats_summary.csv
│  ├─ rare_top150_from_seg_stats.csv
│  ├─ v4_FE_candidate_list.json
├─ notebooks/
│  ├─ v3_5_model_code.ipynb
└─ data/   (로컬 전용, gitignore 대상)
   └─ df_master_preprocessed_v1.parquet
8.2 재현을 위한 필수 파일
data/df_master_preprocessed_v1.parquet

features/top150_final.parquet

notebooks/v3_5_model_code.ipynb

(옵션) features/v3_5_feature_list.csv — 피처 확인용

8.3 실행 순서 (v3.5 기준)
Jupyter / VSCode에서 notebooks/v3_5_model_code.ipynb 열기

위에서부터 순서대로 셀 실행

데이터 로드

Train/Val split

class_weight 계산

v3.5 Feature Engineering (Top150 + FE6)

XGBoost 학습 (max_depth=6, n_estimators=500)

Validation 결과에서:

Macro F1 ≈ 0.53 근처

Segment별 Precision/Recall/Confusion Matrix 확인

9. 최종 목표 정리
모델링 관점

강한 불균형(5클래스, 희귀 클래스 포함) 상황에서
실무적으로 설득력 있는 세그먼트 분류 모델 설계

v3.5를 안정된 베이스라인으로 두고,

v4 이후에는 희귀 세그먼트(0,1)의 인식력 개선에 집중

비즈니스/마케팅 관점

Segment 2/3/4:

활동 수준·이용채널·리스크 수준에 따라 상품 추천·혜택 설계

Segment 0/1:

“과거 고활동 → 현재 비활동” 희귀 고객군

→ 휴면고객 리마케팅, 리스크 관리, 한도 재조정 전략 등으로 확장 가능

프로젝트 산출물 관점 (팀 공유용)

EDA 리포트 (이 README + 상세 노트북)

v2/v3/v3.5 실험 코드 및 결과

v4 Feature 후보 JSON (v4_FE_candidate_list.json)

최종 발표/보고서에서 사용할 세그먼트별 프로필 + 전략 시나리오의 기반 데이터

신용카드 고객 세그먼트 분류 프로젝트 보고서

– 극심한 클래스 불균형을 2-Stage Hierarchical Model로 해결 –

1. 서론
1.1 연구 배경

신용카드사는 고객의 이용 패턴을 바탕으로 고객을 여러 세그먼트로 구분하고,
각 세그먼트에 맞는 마케팅·리스크 전략을 수립한다.
특히 고한도·고가치지만 현재는 거의 사용하지 않는 휴면 고객군은
재활성화·한도 관리 관점에서 매우 중요한 타깃이다.

본 프로젝트의 데이터에서는 Segment 0, 1이 이러한 희귀 고객군에 해당하며,
전체 40만 명 중 약 **0.05%**에 불과할 정도로 비율이 극단적으로 낮다.
일반적인 다중 분류 모델로는 이들을 안정적으로 식별하기 어렵기 때문에,
클래스 불균형을 구조적으로 해결하는 접근이 필요했다.

1.2 프로젝트 목표

본 프로젝트의 목표는 다음과 같다.

신용카드 고객을 **5개 Segment(0~4)**로 분류하는 머신러닝 모델 구축

특히 **희귀 세그먼트(0, 1)**에 대한 탐지 성능(Rare F1, Recall)을
실무적으로 활용 가능한 수준까지 끌어올리는 것

모델 구조와 Feature Engineering을 통해
휴면 고객 리마케팅, 한도 재조정 등 비즈니스 활용이 가능한 세그먼트 체계를 설계하는 것

2. 데이터 및 문제 정의
2.1 데이터 개요

총 샘플 수: 400,000명

피처 수: 165개

Hybrid Top150 피처

v4 Feature Engineering(이하 v4 FE) 파생변수 15개

타깃 변수: Segment (0, 1, 2, 3, 4)

데이터는 고객 기본 정보, 이용금액/건수, 청구·입금, 한도·잔액, 포인트/마일리지, 날짜/경과일 등
다양한 금융·행태적 정보를 포함한다.

2.2 클래스 불균형 구조

Train/Validation 분할 후 세그먼트 분포는 아래와 같다.

Segment	Train	Val	비율	특징
0	130	32	0.04%	극희귀, 고가치 휴면
1	19	5	0.006%	극희귀, 장기 비활성
2	17,012	4,253	5.3%	프리미엄
3	46,565	11,642	14.5%	일반 활성
4	256,270	64,068	80.1%	대다수

문제의 핵심은 다음과 같다.

Segment 0, 1은 데이터 수가 극도로 부족하며,
전체 비중은 매우 낮지만 고한도·고가치 고객일 가능성이 높다.

반대로 Segment 4는 전체의 80% 이상을 차지하는 초다수 클래스다.

따라서, 단순한 5-class Direct Classification으로는
Rare 클래스의 F1 및 Recall이 거의 0에 가깝게 나온다.

3. 모델링 전략
3.1 v3.5 Baseline: 5-class Direct Classification

먼저, 전체 Segment를 한 번에 분류하는 v3.5 Baseline을 구축하였다.

입력 피처: Top150 + v3_FE6 → 총 156개

모델: XGBoost (multi:softprob, depth=6, class_weight 적용)

결과(Macro F1, Val): 약 0.5313

이 Baseline은 Segment 2/3/4에 대해서는
괜찮은 수준의 F1을 보였지만, Rare(0, 1)에 대해서는
데이터 부족으로 인해 예측력 확보에 실패했다.

3.2 v4 전략: 2-Stage Hierarchical Model

Rare 문제를 해결하기 위해 모델 구조 자체를 재설계했다.
v4에서는 다음과 같은 2-Stage Hierarchical Model을 도입하였다.

Stage 1: Binary Classification

타깃: Rare(0+1) vs Others(2+3+4)

역할: 전체 고객 중 Rare 후보군을 우선적으로 식별

Stage 2-B: Multi-class Classification

입력: Stage 1에서 Others로 분류된 고객

타깃: Segment 2, 3, 4

역할: 데이터가 충분한 구간에 대해 안정적인 세분화 수행

이 구조를 통해, 희귀 세그먼트 문제를
“희귀 vs 비희귀” 문제와 “비희귀 내 세분화” 문제로 나누어 접근했다.

4. Feature Engineering 및 시스템 구성
4.1 v4 Feature Engineering (18개)

Rare vs Others의 행동 패턴 차이를 강조하기 위해,
v4에서는 도메인 지식 기반 파생변수 18개를 추가 설계하였다.

(1) Recency 지표 (3개)

v4_last_use_gap_CA: CA 영역 마지막 이용 후 경과일

v4_last_use_gap_card_all: 전체 카드 기준 마지막 이용 후 경과일

v4_first_to_last_gap: 가입 ~ 마지막 이용까지의 기간

→ **“언제까지 쓰다가 끊었는지”**를 직접적으로 반영하는 변수들이다.

(2) 효율성 지표 (2개)

v4_limit_to_usage_ratio_R12M: 한도 대비 12개월 사용 비율

v4_balance_to_usage_ratio: 평잔 대비 6개월 사용 비율

→ 한도·잔액 대비 실제 사용 정도를 보여주는 지표로,
슬리핑 하이리밋(sleeping high-limit) 고객을 식별하는 데 유용하다.

(3) 변화 패턴 (2개)

v4_bill_drop_R6_to_R3: 6M → 3M 청구금액 감소율

v4_usage_volatility_R3_R6_R12: 구간별 사용금액 변동성

→ 사용 패턴의 급격한 감소·변동을 포착하여
“최근에 사용이 줄었는가?”를 반영한다.

(4) Flag 지표 (5개)

v4_recent_zero_usage_flag: 최근 3개월 완전 미사용 여부

v4_long_inactive_high_limit_flag: 장기 미사용 + 고한도 플래그

v4_arrears_recent_flag: 최근 연체(30일+) 여부

v4_cardloan_cleanup_flag: 카드론 정리 후 비활성 패턴 여부

v4_lifestyle_auto_payment_flag: 자동납부 비활성 여부

→ Rare 세그먼트가 가질 법한 “이상 행동 패턴”을 명시적으로 표현한다.

(5) 라이프스타일 지표 (3개)

v4_point_activity_intensity: 포인트 활동 강도

v4_travel_mileage_activity: 마일리지 관련 활동 정도

v4_online_offline_usage_ratio_R6M: 온라인 vs 오프라인 비율

(6) 종합 지표 (3개, 최적화 버전)

v4_dormant_score: 종합 휴면 점수 (0~5)

v4_high_value_dormant: 고가치 휴면 고객 여부

v4_usage_drop_ratio: 6M vs 12M 사용 감소율

4.2 시스템 및 리포지토리 구조

프로젝트 디렉터리는 다음과 같이 구성하였다.

data/: 전처리된 마스터 데이터(Parquet) 저장

features/: Top150, v3.5 피처 목록, Rare 특화 피처, v4 FE 정의 등

notebooks/: v3.5 Baseline, v4 Two-Stage, v4 최적화 버전 노트북

models/: 학습 완료된 모델(.pkl)과 설정(.json)

docs/: README, Quickstart, Overflow Fix, 실행 가이드, 프로젝트 요약 등 문서

이 구조를 통해 **재현 가능성(reproducibility)**과
협업 시 가독성을 모두 고려하였다.

5. 실험 및 성능 분석
5.1 v3.5 Baseline 성능

접근: 5-class Direct Classification

피처: Top150 + v3_FE6 (총 156개)

모델: XGBoost (depth=6, class_weight)

Macro F1 (Val): ~0.5313

특징

Segment 2/3/4: 0.7~0.9 수준의 안정적인 F1

Segment 0: 일부 예측 가능

Segment 1: 데이터 수가 너무 적어 사실상 예측 불가

→ 전체 Macro F1은 그럭저럭 나오지만,
희귀 세그먼트 관점에서는 활용이 어려운 수준이다.

5.2 v4 Two-Stage 기본 모델 성능

접근: 2-Stage Hierarchical Classification

Stage 1: Rare(0+1) vs Others(2+3+4)

Stage 2-B: Others 내부 2/3/4 분류

피처: Top150 + v4 FE 15개 (총 165개)

결과

Rare F1: ~0.30 수준까지 개선

Final Macro F1: ~0.53 (v3.5와 유사)

→ 구조만 바꿔도 Rare 성능이 유의미하게 상승하는 것을 확인했다.

5.3 v4 최적화 모델 성능

최적화 버전에서는 다음을 추가로 적용하였다.

SMOTE를 활용한 Rare Oversampling

RandomizedSearchCV 기반 Hyperparameter Tuning

XGBoost + LightGBM 앙상블

Threshold 조정(≈0.35)으로 F1 최대화

v4 FE 18개까지 확장

Stage 1: Rare vs Others

Rare F1: 약 0.52

Rare Recall: 56.8%

Macro F1(Stage 1): 약 0.76

Stage 2-B: 2/3/4 분류

Seg 2 F1: 약 0.70

Seg 3 F1: 약 0.87

Seg 4 F1: 약 0.98

Macro F1(Stage 2-B): 약 0.85

최종 결과(5-class 관점)

Final Macro F1: 약 0.58+

기존 v3.5 대비 Macro F1 약 +0.05p 향상

Rare F1은 0.1 → 0.52 수준으로 약 4배 이상 개선

5.4 요약
지표	v3.5	v4 최적화	변화
Rare F1	~0.1	~0.52	+420% 개선
Final Macro F1	~0.53	~0.58	성능 상승
Rare Recall	~0%	~56.8%	“실질적 탐지 가능” 수준

정리하면, v4는 단순한 성능 튜닝이 아니라
모델 구조(2-Stage) + Feature Engineering + Oversampling + Threshold 최적화를 통해
희귀 세그먼트 탐지력을 본질적으로 끌어올린 버전이라고 볼 수 있다.

6. 주요 기술 이슈 및 해결 방법

프로젝트 진행 중 다음과 같은 기술적 문제를 겪었고, 이를 해결하였다.

6.1 날짜 오버플로우 (OutOfBoundsTimedelta)

이슈: rv최초시작후경과일 컬럼에 99999999와 같은 극단값이 존재하여
to_timedelta 변환 시 overflow 발생

해결:

상한값을 **50년(18,250일)**로 클리핑

청크 단위로 나누어 안전하게 계산

6.2 XGBoost 레이블 제약

이슈: XGBoost는 multi-class 레이블이 0부터 시작하는 연속 정수여야 함
(예: [2,3,4]만 존재하는 경우 에러 발생)

해결:

Stage 2-B 타깃인 2/3/4를 0/1/2로 매핑 후 학습

예측 후 reverse mapping으로 복구

6.3 Stage 간 샘플 수 불일치

이슈: 학습 시와 최종 예측 시 Stage 2-B에 들어가는 샘플 수가 달라
boolean mask assignment 에러 발생

해결:

최종 예측 시, Stage 1에서 Others로 나온 전체 샘플에 대해
동일한 Stage 2-B 모델로 다시 예측하는 방식으로 통일

6.4 NaN 레이블 및 변수명 오타

Rare가 Others로 잘못 넘어가면서 레이블 매핑에 실패해 NaN이 생기는 이슈

대소문자 차이(CHUNK_size vs CHUNK_SIZE)로 인한 NameError 등

모두 조건 필터 정교화 및 변수명 정리로 해결

이러한 이슈 해결 과정은 OVERFLOW_FIX.md 및 관련 문서에 별도로 정리하였다.

7. 비즈니스 시사점
7.1 Rare 세그먼트 (0+1)

특징

과거에는 고액/고빈도 사용 이력이 있으나,
최근에는 사용이 거의 없는 휴면 고객

한도가 높고 평잔도 높은데, 실제 이용은 적은 경우 존재
→ High-value Dormant 고객 가능성이 큼

활용 방안

휴면 고객 리마케팅 타깃

맞춤형 재활성화 프로모션

VIP 혜택 재제공, 마일리지/포인트 보너스 등

한도·리스크 관리

고한도 미사용 고객에 대한 한도 재조정 시뮬레이션

이탈 방지 및 리스크 관리 지표로 활용

7.2 Others (2/3/4)

Segment 2: 프리미엄 고객 → 마일리지·프리미엄 혜택 중심 전략

Segment 3: 일반 활성 고객 → 대중적인 할인·적립형 상품

Segment 4: 저활동/대다수 고객 → 활성화 유도, 기본 서비스 제공

결과적으로, 본 모델은
세그먼트별 차별화 전략을 설계할 수 있는 최소 단위를 제공한다고 볼 수 있다.

8. 결론 및 향후 과제
8.1 결론

본 프로젝트에서는 극심한 클래스 불균형 상황에서
신용카드 고객을 5개 세그먼트로 분류하는 모델을 설계·구현하였다.

v3.5 Baseline은 5-class Direct Classification 기반으로
Macro F1 ~0.53 수준의 성능을 달성했지만, Rare 세그먼트에 대한 예측력은 매우 낮았다.

v4에서는 2-Stage Hierarchical Model + 도메인 기반 Feature Engineering + SMOTE + Threshold 최적화를 통해
Rare F1을 약 0.52까지 끌어올렸고, Macro F1 역시 ~0.58 수준으로 개선하였다.

이를 통해, 희귀하지만 비즈니스적으로 중요한 고객군을 실질적으로 탐지 가능한 수준까지 성능을 향상시켰다.

8.2 향후 개선 방향 (v5 후보)

Stage 2-A 구현

Rare 내부에서 Segment 0 vs 1을 추가로 구분하는 모델 설계

Tabular Deep Learning 도입

TabNet, AutoInt 등의 모델을 활용해 FE 의존도를 줄이는 실험

Focal Loss / Cost-sensitive Learning

희귀 클래스에 더 강한 페널티를 주는 손실 함수 적용

실시간 스코어링 환경 구축

FastAPI + Docker를 사용한 실시간 예측 API 개발

Explainability 강화

SHAP, LIME 등을 활용해 세그먼트별 주요 결정 요인 시각화
