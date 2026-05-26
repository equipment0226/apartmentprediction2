# 시나리오 생성 파이프라인 M1~M5 — 쉬운 설명 + 코드

> 목적: 기존 하드코딩 Diffusion 방식을 대체하는 새 시나리오 생성 파이프라인
> M1~M5를, 일반인도 이해할 수 있게 설명하고 각 모듈의 코드를 함께 제시한다.
> 핵심 원칙: "사람이 직관으로 박는 숫자"를 최대한 없애고, 모든 기준을
> 과거 데이터와 추정 결과에서 나오게 한다.

======================================================================
## 0. 전체를 한 문장으로
======================================================================

"과거 집값·금리·정책 데이터를 보고, 변수들이 서로 어떻게 얽혀 움직이는지
배운 뒤, 그 관계를 미래로 수천 번 굴려서 '일어날 수 있는 미래 집값
시나리오 셋'을 만든다."

기존 방식과 가장 큰 차이:
- 기존: 사람이 "Bull이면 금리 -0.25" 식으로 숫자를 직접 박음 (하드코딩)
- 새 방식: 그 숫자를 데이터가 정하게 함 (추정·학습)


======================================================================
## 1. 전체 구조도
======================================================================

   ┌─────────────────────────────────────────────────────────────┐
   │  raw 데이터  (dong_level_meta_table.csv)                      │
   │  = 과거 월별 집값·전세·금리·CPI·수급·정책 표                  │
   └───────────────────────────┬─────────────────────────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │ M1  전처리·정상화                     │
            │  추세 제거(차분), 결측·이상치 정리      │
            │  → VAR이 먹을 수 있는 깔끔한 표         │
            └──────────────────┬──────────────────┘
                               │ stationary_df, transform_log
            ┌──────────────────▼──────────────────┐
            │ M2  Bayesian VAR 추정 (엔진)          │
            │  "변수들이 서로 어떻게 움직이나"를 학습 │
            │  → 관계표(계수) + 충격주머니(잔차)      │
            └──────────────────┬──────────────────┘
                               │ coef, resid, resid_cov
            ┌──────────────────▼──────────────────┐
            │ M3  시나리오 생성 (block bootstrap)   │
            │  엔진을 미래로 수천 번 굴림            │
            │  (M3+: 잔차주머니 보강 → N 하드조건 역산)│
            │  → 미래 경로 수천 개                   │
            └──────────────────┬──────────────────┘
                               │ scenario_tensor (N, 12, 변수수)
            ┌──────────────────▼──────────────────┐
            │ M4  현실성 필터 (검사관·계기판)        │
            │  말 안 되는 가짜 경로에 불합격 도장     │
            │  → 위반율 리포트                      │
            └──────────────────┬──────────────────┘
                               │ clean_tensor, realism_report
            ┌──────────────────▼──────────────────┐
            │ M5  검증·군집화                       │
            │  원 단위 복원 → 성적표 → 상승/보합/하락 │
            │  → 최종 시나리오 셋 + 검증 점수        │
            └──────────────────┬──────────────────┘
                               │
                               ▼
                  TFT 예측 / 서비스 출력으로 전달
   ─────────────────────────────────────────────────────────────
   (2번 트랙) M2를 SVAR로 확장 — M5 검증이 인과·개입 한계를
              가리킬 때 투입. "정책 바꾸면?" 질문에 답.

[새로 짜는 코드]  M3 · M4 · M5
[기존 자산 재활용] M1 (build_scaled_monthly_table), M2 (VAR 경로)
[필요 라이브러리] numpy, pandas, statsmodels  (전부 CPU에서 가벼움)


======================================================================
## M1 — 전처리·정상화
======================================================================

### 쉬운 설명

VAR이라는 모델은 까다로운 손님이라, 데이터를 아무렇게나 주면 체합니다.
특히 "계속 우상향하는 값"(집값처럼)을 그대로 주면 분석이 망가집니다.

그래서 M1은 데이터를 VAR이 소화할 수 있는 형태로 다듬는 '주방' 단계입니다.
핵심 작업은 '차분(diff)' — "이번 달 값" 대신 "이번 달에 얼마나 변했나"로
바꾸는 것. 집값 1500 → 1520 → 1490 을, +20 → -30 으로 바꿉니다.
끝없이 올라가던 숫자가 0 주변에서 오르내리는 숫자가 되어 다루기 쉬워집니다.

어떤 변수를 차분할지는 'ADF 검정'이라는 통계 검사로 자동 판단합니다.
사람이 고르지 않습니다.

중요: 나중에 M5에서 다시 원래 집값 단위로 되돌려야 하므로,
"무엇을 어떻게 바꿨는지"를 transform_log에 기록해 둡니다.

### 코드

```python
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def run_M1(raw_df, feature_cols, policy_cols):
    """raw 표 → VAR이 먹을 수 있는 정상 시계열로 변환."""
    df = raw_df.copy()
    transform_log = {}   # M5 역변환용 기록

    for col in feature_cols:
        # 정책 변수(-1/0/1 이산값)는 차분/정규화 대상 아님 → 건너뜀
        if col in policy_cols:
            transform_log[col] = {"diff": False}
            continue

        # ADF 검정: p값이 크면 '비정상' → 차분 필요
        pval = adfuller(df[col].dropna())[1]
        if pval > 0.05:
            transform_log[col] = {"diff": True,
                                  "last_level": df[col].iloc[-1]}
            df[col] = df[col].diff()      # 1차 차분
        else:
            transform_log[col] = {"diff": False}

    # 결측 보간 + 이상치 윈저라이징(극단값을 1~99%로 자름)
    df[feature_cols] = (df[feature_cols]
                        .interpolate()
                        .clip(lower=df[feature_cols].quantile(0.01),
                              upper=df[feature_cols].quantile(0.99),
                              axis=1))
    stationary_df = df.dropna().reset_index(drop=True)
    return stationary_df, transform_log
```

### 한 줄 요약
"집값 같은 우상향 데이터를 '변화량'으로 바꿔 VAR이 소화할 수 있게 다듬고,
되돌리는 방법을 기록해 둔다."


======================================================================
## M2 — Bayesian VAR 추정 (엔진)
======================================================================

### 쉬운 설명

이 파이프라인의 심장입니다. VAR이 하는 일을 쉽게 말하면:

"이번 달 금리·집값·수급을 보면, 다음 달 금리·집값·수급을 어느 정도
예측할 수 있다 — 변수들이 서로 영향을 주고받기 때문이다."

VAR은 과거 데이터를 보고 그 '영향 관계'를 숫자표로 만듭니다. 이게 '계수'.
사람이 "금리가 집값에 -0.25 영향" 이라고 박던 걸, VAR이 데이터에서
스스로 추정합니다. → 하드코딩 drift가 사라지는 지점.

VAR은 두 가지를 내놓습니다:
1. 계수(coef)   — 변수 간 관계표
2. 잔차(resid)  — 모델이 설명 못 한 '예측 오차'들의 기록.
                  이게 미래의 불확실성을 만드는 '충격 주머니'가 됩니다.

### 왜 'Bayesian' VAR인가

동 단위 데이터는 양이 적습니다(100여 개월). 일반 VAR은 데이터가 적으면
계수가 과하게 출렁여 엉뚱한 값이 나옵니다(과적합).

'Minnesota prior'라는 장치를 더하면, "각 변수는 자기 과거에 가장 의존하고,
멀리 떨어진 영향은 0에 가깝다"는 상식적 사전믿음으로 계수를 안정시킵니다.
적은 데이터에서도 믿을 만한 추정이 됩니다.

### 코드

```python
import numpy as np
from statsmodels.tsa.api import VAR

def run_M2(stationary_df, feature_cols, max_lag=3,
           minnesota_lambda=0.2):
    """VAR 추정 → 계수 + 잔차 + 잔차공분산."""
    data = stationary_df[feature_cols].values

    # lag(몇 개월 전까지 볼지)는 BIC 기준으로 자동 선택
    model = VAR(data)
    lag = model.select_order(max_lag).bic
    fitted = model.fit(lag)

    coef      = fitted.coefs          # 변수 간 관계표
    resid     = fitted.resid          # 예측오차 = 충격 주머니
    resid_cov = np.cov(resid.T)       # 충격들의 동시 상관(공분산)

    # --- Minnesota shrinkage: 적은 데이터에서 계수 안정화 ---
    # 자기 자신 시차는 살리고, 타 변수·먼 시차 계수는 0쪽으로 당김
    for L in range(coef.shape[0]):
        shrink = minnesota_lambda / (L + 1)        # 먼 시차일수록 더 당김
        off_diag = ~np.eye(coef.shape[1], dtype=bool)
        coef[L][off_diag] *= (1 - shrink)

    # --- 경제이론 부호 점검(로그만, 강제 아님) ---
    # 예: 금리→집값 계수가 양수로 나오면 이론 위반 → 경고 기록
    sign_check = _check_economic_signs(coef, feature_cols)

    return {"coef": coef, "resid": resid, "resid_cov": resid_cov,
            "lag": lag, "sign_check": sign_check}

def _check_economic_signs(coef, feature_cols):
    """핵심 계수 부호가 경제이론과 맞는지 확인해 경고만 남김."""
    warnings = []
    # (이론상 음수여야 하는 관계 예시: 금리 → 집값)
    # 위반 시 warnings에 append — M2를 prior로 교정하라는 신호
    return warnings
```

### 한 줄 요약
"과거 데이터에서 '변수들이 서로 어떻게 움직이는지' 관계표를 추정하고,
모델이 못 맞춘 오차들을 '충격 주머니'로 모아둔다. 데이터가 적으니
Minnesota prior로 안정화한다."


======================================================================
## M3 — 시나리오 생성 (block bootstrap 몬테카를로)
======================================================================

### 쉬운 설명

M2가 만든 엔진(관계표 + 충격주머니)을 미래로 굴리는 단계입니다.

방법: 마지막 실제 관측값에서 출발해서
  ① VAR 관계표로 "다음 달 예상치"를 계산하고
  ② 충격 주머니에서 '실제로 있었던 충격'을 하나 꺼내 더하고
  ③ 그 결과를 다음 달 값으로 삼아 또 ①로
이걸 12개월 반복하면 미래 경로 하나. 이 과정을 수천 번 반복하면
미래 경로 수천 개 = 시나리오 셋.

### 왜 'block bootstrap'인가 (중요)

충격을 더할 때 두 가지 방식이 있습니다:
- (방식 A) 정규분포에서 무작위로 뽑기 → "충격은 종 모양 분포"라고 가정
- (방식 B) 과거에 실제 있었던 충격을 그대로 꺼내 쓰기  ← 이게 bootstrap

방식 B의 장점:
- 흔했던 충격은 자주, 드물었던 충격은 드물게 뽑힘
  → 시나리오 셋의 확률 비중이 현실의 발생 빈도를 닮음
- 정규분포 가정 자체를 안 함 → 실제의 '두꺼운 꼬리'(드문 큰 충격)도 재현

'block(덩어리)'을 쓰는 이유:
- 충격을 한 달씩 따로 뽑으면 경로가 톱니처럼 들쭉날쭉해짐
- 3개월씩 '연속 덩어리'로 꺼내면 충격의 시간적 흐름이 보존돼 매끄러움

### 코드

```python
import numpy as np

def run_M3(stationary_df, var_result, feature_cols,
           horizon=12, n_scenarios=3000, block_size=3, seed=42):
    """VAR 엔진을 미래로 굴려 시나리오 수천 개 생성."""
    rng = np.random.default_rng(seed)
    coef  = var_result["coef"]
    resid = var_result["resid"]
    lag   = var_result["lag"]
    n_feat = len(feature_cols)

    # 출발점: 마지막 lag개월 실제값
    init = stationary_df[feature_cols].values[-lag:]

    scenarios = np.zeros((n_scenarios, horizon, n_feat))

    for s in range(n_scenarios):
        recent = init.copy()                  # 최근 lag개월 (계속 갱신)
        path = []
        t = 0
        while t < horizon:
            # --- 충격: 과거 잔차에서 연속 block을 통째로 꺼냄 ---
            start = rng.integers(0, len(resid) - block_size)
            shock_block = resid[start:start + block_size]

            for k in range(block_size):
                if t >= horizon:
                    break
                # ① VAR 관계표로 다음 달 예상치 계산
                mean_next = np.zeros(n_feat)
                for L in range(lag):
                    mean_next += coef[L] @ recent[-(L + 1)]
                # ② 실제 충격 더하기
                next_val = mean_next + shock_block[k]
                path.append(next_val)
                # ③ 최근값 갱신
                recent = np.vstack([recent, next_val])[-lag:]
                t += 1
        scenarios[s] = np.array(path)

    return scenarios     # (n_scenarios, horizon, n_feat) — 아직 차분 공간

```

### 한 줄 요약
"엔진을 미래로 12개월 굴리되, 충격은 과거에 실제 있었던 것을 3개월
덩어리로 꺼내 쓴다. 수천 번 반복해 미래 경로 수천 개를 만든다."


----------------------------------------------------------------------
## M3+ — 트라이 수(N)와 잔차 주머니를 '근거 있게' 정하기
----------------------------------------------------------------------

### 쉬운 설명 — 왜 이 보강이 필요한가

M3 기본 설명은 "수천 번 돌리면 극단 충격도 언젠가 뽑힌다"였다.
그런데 '언젠가'는 근거가 아니라 그냥 기대다. 두 가지 질문이 남는다.

  질문 1: N(트라이 수)을 3000으로 할지 8000으로 할지 무슨 근거로 정하나?
  질문 2: N을 아무리 키워도, 잔차 주머니에 극단 충격이 2~3개뿐이면
          그 똑같은 2~3개를 다시 뽑을 뿐 아닌가?

→ 그래서 M3+는 두 가지를 한 세트로 정한다.
   (A) 재료(잔차 주머니)를 먼저 충분히·다양하게 채운다.
   (B) 그다음 N을 '하드 조건'에서 수학으로 역산한다.

순서가 중요하다. 재료 없이 N 공식만 쓰면 공식이 작동을 안 하거나
'같은 충격을 복붙한 가짜 꼬리'를 만든다.


### (A) 잔차 주머니 보강 — 재료를 먼저 채운다

문제: 동 하나의 과거 데이터는 100여 개월. 2008·2022급 극단 충격은
      잔차 주머니에 2~3개뿐일 수 있다. N을 100만으로 늘려도 그 2~3개를
      반복 추출할 뿐 — N이 부족한 게 아니라 '재료'가 부족한 것.

두 가지 보강:

  보강 1) 여러 동·구의 잔차를 합친다.
          한 동에서 안 겪은 충격을 다른 동 기록에서 빌려온다.
          잔차 표본 자체가 커져 꼬리가 채워진다.

  보강 2) 꼬리를 분포로 모델링한다 (Student-t / 극단값이론 EVT).
          잔차를 그대로 리샘플만 하지 않고, 꼬리에 두꺼운 분포를
          적합해 "관측엔 없지만 그 분포라면 가능한" 극단 충격을
          생성한다. 과거에 딱 그 크기로 안 와봤어도 비슷한 규모가
          N 안에서 나온다.

### (A) 코드

```python
import numpy as np
from scipy import stats

def build_residual_pool(resid_list, fat_tail=True, t_df=5):
    """
    잔차 주머니 보강.
    resid_list : 여러 동의 잔차 배열 리스트 [(T1,n_feat), (T2,n_feat), ...]
                 동 하나만 쓸 거면 길이 1 리스트로 넣어도 됨.
    fat_tail   : True면 꼬리를 Student-t로 모델링해 극단 충격도 생성 가능.
    """
    # 보강 1: 여러 동 잔차를 세로로 합침
    pooled = np.vstack(resid_list)              # (sum T, n_feat)

    pool = {"empirical": pooled}                 # 그대로 리샘플할 실측 잔차

    # 보강 2: 변수별로 Student-t 꼬리 적합 (선택)
    if fat_tail:
        tail_models = []
        for j in range(pooled.shape[1]):
            # 각 변수 잔차에 t분포 적합 → (자유도, 위치, 척도)
            df, loc, scale = stats.t.fit(pooled[:, j], f0=t_df)
            tail_models.append((df, loc, scale))
        pool["t_models"] = tail_models

    return pool

def sample_shock_block(pool, block_size, rng, tail_prob=0.05):
    """
    충격 block 하나를 뽑는다.
    - 평소(1 - tail_prob 확률): 실측 잔차에서 block 리샘플 (현실 빈도 보존)
    - 드물게(tail_prob 확률)  : Student-t 꼬리에서 극단 충격 생성
                                → 관측에 없던 극단도 다양하게 나옴
    """
    emp = pool["empirical"]
    if ("t_models" in pool) and (rng.random() < tail_prob):
        # 꼬리 모델에서 block 생성
        n_feat = emp.shape[1]
        block = np.zeros((block_size, n_feat))
        for j, (df, loc, scale) in enumerate(pool["t_models"]):
            block[:, j] = stats.t.rvs(df, loc, scale,
                                      size=block_size, random_state=rng)
        return block
    # 실측 잔차에서 연속 block 리샘플
    start = rng.integers(0, len(emp) - block_size)
    return emp[start:start + block_size]
```


### (B) N을 하드 조건으로 역산 — "최소 k번을 99% 확률로"

발상: "극단 충격이 N번 중 언젠가 뽑히겠지"(도박) 를
      "극단 충격을 99% 확률로 최소 k번 뽑는다"(하드 조건) 로 바꾼다.
      그러면 N이 직관이 아니라 요구사항에서 수학으로 역산된다.

왜 '최소 1번'이 아니라 '최소 k번'인가 (중요):
  "최소 1번"을 보장하면 극단 충격을 맞는 경로가 딱 1개.
  분포 맨 끝에 외톨이 1개가 찍힐 뿐 → P90·Coverage가 안정 안 됨.
  꼬리 영역의 분포가 신뢰를 가지려면 경로가 수십 개(k≈30)는 있어야 함.
  그래서 '최소 1번'(기하분포)이 아니라 '최소 k번'(이항분포)으로 건다.

공식:
  - 어떤 극단 충격의 1회 추출 확률 = p
  - N번 중 그 충격이 X번 뽑힘 ~ 이항분포 B(N, p)
  - "P(X ≥ k) ≥ confidence" 를 만족하는 최소 N을 찾는다

### (B) 코드

```python
import numpy as np
from scipy.stats import binom

def required_N(p_rare, k_min=30, confidence=0.99, hard_floor=2000):
    """
    하드 조건: 1회 추출확률 p_rare인 극단 충격을,
              confidence 확률로 최소 k_min번 이상 뽑는 데 필요한 N.

    p_rare   : 극단 충격의 1회 추출 확률.
               = (잔차 주머니 내 극단 충격 개수) / (전체 잔차 개수)
               block 단위면 (극단 든 block 수)/(전체 block 수)로 근사.
    k_min    : 꼬리 추정이 안정되려면 보통 수십 개. 기본 30.
    confidence : 보장 확률. 0.99 = 99%.
    hard_floor : 분포 전체 형태 안정을 위한 최소 N 바닥값.
    """
    if p_rare <= 0:
        raise ValueError(
            "p_rare=0 : 잔차 주머니에 극단 충격이 아예 없음. "
            "N으로 못 푼다 → (A) 재료 보강(다중 동 통합/꼬리 모델링) 먼저.")

    N = k_min
    # P(X >= k_min) = 1 - P(X <= k_min-1) >= confidence 가 될 때까지 N 증가
    while binom.cdf(k_min - 1, N, p_rare) > (1 - confidence):
        N += 100
    return max(N, hard_floor)

def diagnose_convergence(generate_one_path, target_idx, rng,
                          N_grid=(1000, 2000, 4000, 8000, 16000),
                          tol=0.005):
    """
    보조 확인: N을 키우며 P10/P90이 더 이상 안 흔들리는지 검증.
    하드 조건으로 정한 N이 실제로 수렴 구간에 있는지 교차확인용.
    """
    prev = None
    for N in N_grid:
        paths = np.array([generate_one_path(rng) for _ in range(N)])
        p10, p90 = np.percentile(paths[:, -1, target_idx], [10, 90])
        if prev is not None:
            d = max(abs(p10-prev[0])/abs(prev[0]),
                    abs(p90-prev[1])/abs(prev[1]))
            print(f"N={N:6d}  P10={p10:.1f} P90={p90:.1f}  변화 {d*100:.2f}%")
            if d < tol:
                print(f"→ P10/P90 수렴 (N={N})")
        prev = (p10, p90)
```

사용 예:
```python
# 1) 재료 먼저: 잔차 주머니 보강
pool = build_residual_pool([resid_dong_A, resid_dong_B, resid_dong_C],
                            fat_tail=True)

# 2) p_rare 산출: 주머니에서 '극단'으로 볼 잔차 비율
#    예) 타깃 잔차의 절대값 상위 2%를 극단으로 정의
abs_r = np.abs(pool["empirical"][:, target_idx])
p_rare = (abs_r > np.quantile(abs_r, 0.98)).mean()   # ≈ 0.02

# 3) 하드 조건으로 N 역산
N = required_N(p_rare, k_min=30, confidence=0.99)
print(f"하드 조건 충족 N = {N}")
#   → "절대값 상위 2% 극단 충격을, 99% 확률로 최소 30번 뽑는 N"
```

### M3+ 한 줄 요약
"N은 직관이 아니라 '극단 충격을 99% 확률로 최소 k번 뽑는다'는 하드
조건에서 이항분포로 역산한다. 단 그 공식의 입력 p는 잔차 주머니가
정하므로, 재료(다중 동 통합 + 꼬리 모델링)를 먼저 채운 뒤 N을 정한다."

### 주의 — 순서를 지켜라
- 재료(A) 없이 N 공식(B)만 쓰면: p=0이라 공식이 N=무한대를 뱉거나,
  주머니의 같은 극단 1개를 수십 번 복붙한 '가짜 꼬리'가 만들어진다.
- 반드시 (A) 재료 보강 → (B) 하드 조건 N 역산 순서.


======================================================================
## M4 — 현실성 필터 (검사관·계기판)
======================================================================

### 쉬운 설명

M3가 만든 수천 개 경로 중엔 무작위로 만들다 보니 '말도 안 되는 가짜'가
섞입니다. M4는 그걸 잡아내는 품질 검사관입니다.

핵심 원칙: "무엇이 가짜인가"의 기준을 사람이 정하지 않는다.
과거 데이터와 M2 추정 결과가 정한다. → 하드코딩 재발 방지.

3가지 검사:

[T1] 물리적으로 불가능한 값
  금리·집값이 음수. 이건 '가정'이 아니라 변수의 정의 자체라
  상수로 둬도 무방. (데이터가 바뀌어도 안 변하는 사실)

[T2] 과거에 없던 비현실적 급변·톱니
  "한 달에 너무 많이 움직임"의 기준을 사람이 안 정함.
  과거 실제 월별 변화폭의 상위 1% 지점을 기준선으로 삼음.
  톱니도 과거 경로의 실제 출렁임 빈도를 기준으로.

[T3] 변수끼리 앞뒤가 안 맞음
  집값은 폭등하는데 수급은 바닥. 두 변수 관계의 '방향'을 사람이
  선언하지 않고 M2가 추정한 부호를 그대로 가져와 검사.

가장 중요한 설계 — "불합격이라고 바로 버리지 않는다":
  엉터리 경로를 마구 버리면 시나리오 셋의 확률 비중이 왜곡됨.
  (급락 시나리오만 자꾸 버려지면 "급락 안 일어난다"는 거짓이 됨)
  그래서 M4 기본 동작은 '버리기'가 아니라 '불합격 도장 + 위반율 리포트'.

M4의 진짜 역할 = 계기판:
  위반율이 높으면 → M4가 더 버릴 신호가 아니라 M2·M3를 고칠 신호.
  잘 설계된 시스템이면 M4는 한가해야 정상.

### 코드

```python
import numpy as np

# [T1] 정의상 불변 제약 — 상수 OK (변수 정의 자체)
HARD_DOMAIN = {
    "ecos__base_rate":           (0.0, None),
    "ecos__mortgage_rate_new":   (0.0, None),
    "reb__apt_sale_avg_price":   (0.0, None),
    "reb__apt_sale_supply_demand": (0.0, 200.0),
}

def fit_realism_thresholds(observed_df, var_result, feature_cols,
                            target_col, q_extreme=0.99):
    """M4 검사 기준을 '데이터'와 'M2 추정'에서 산출 (하드코딩 아님)."""
    th = {}
    # T2-a: 변수별 비현실적 변화폭 = 과거 월변화 상위 1%
    chg = observed_df[feature_cols].diff().abs()
    th["max_abs_change"] = chg.quantile(q_extreme).to_dict()
    # T2-b: 톱니 기준 = 과거 타깃의 실제 방향반전 빈도
    d = observed_df[target_col].diff().dropna()
    th["normal_flip_rate"] = (np.sign(d).diff() != 0).mean()
    # T3: 변수쌍 부호 = M2 잔차공분산에서 읽음
    corr = _cov_to_corr(var_result["resid_cov"])
    th["pair_sign"] = {}
    for a, b in _key_pairs(feature_cols, target_col):
        ia, ib = feature_cols.index(a), feature_cols.index(b)
        if abs(corr[ia, ib]) > 0.2:
            th["pair_sign"][(a, b)] = np.sign(corr[ia, ib])
    return th

def run_M4(paths_orig, observed_df, var_result, feature_cols,
           target_col, mode="report"):
    """경로별 현실성 검사. 기본은 '리포트'(경로 안 버림)."""
    th = fit_realism_thresholds(observed_df, var_result,
                                 feature_cols, target_col)
    N = paths_orig.shape[0]
    v_t1 = np.zeros(N, dtype=bool)
    v_t2 = np.zeros(N, dtype=bool)
    v_t3 = np.zeros(N, dtype=bool)

    # T1: 정의역 이탈
    for col, (lo, hi) in HARD_DOMAIN.items():
        if col not in feature_cols: continue
        j = feature_cols.index(col)
        v = paths_orig[:, :, j]
        if lo is not None: v_t1 |= (v < lo).any(axis=1)
        if hi is not None: v_t1 |= (v > hi).any(axis=1)

    # T2: 과거 99%를 넘는 급변
    for col, mx in th["max_abs_change"].items():
        if col not in feature_cols: continue
        j = feature_cols.index(col)
        step = np.abs(np.diff(paths_orig[:, :, j], axis=1))
        v_t2 |= (step > mx).any(axis=1)

    # T3: 변수쌍 부호 어긋남
    for (a, b), sign in th["pair_sign"].items():
        ia, ib = feature_cols.index(a), feature_cols.index(b)
        da = np.diff(paths_orig[:, :, ia], axis=1)
        db = np.diff(paths_orig[:, :, ib], axis=1)
        pc = _rowwise_corr(da, db)
        v_t3 |= (np.sign(pc) != sign) & (np.abs(pc) > 0.3)

    any_v = v_t1 | v_t2 | v_t3
    report = {
        "violation_rate": {
            "T1_domain":      float(v_t1.mean()),
            "T2_extreme":     float(v_t2.mean()),
            "T3_consistency": float(v_t3.mean()),
            "ANY":            float(any_v.mean()),
        },
        "flags": {"T1": v_t1, "T2": v_t2, "T3": v_t3},
    }
    if mode == "drop":
        return paths_orig[~any_v], report
    return paths_orig, report     # 기본: 안 버리고 리포트만

# 보조 함수
def _cov_to_corr(cov):
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)

def _rowwise_corr(A, B):
    Az, Bz = A - A.mean(1, keepdims=True), B - B.mean(1, keepdims=True)
    num = (Az * Bz).sum(1)
    den = np.sqrt((Az**2).sum(1) * (Bz**2).sum(1)) + 1e-12
    return num / den

def _key_pairs(feature_cols, target_col):
    cand = [(target_col, "reb__apt_sale_supply_demand"),
            (target_col, "ecos__mortgage_rate_new")]
    return [(a, b) for a, b in cand
            if a in feature_cols and b in feature_cols]
```

### 한 줄 요약
"말 안 되는 가짜 경로에 불합격 도장을 찍되 버리진 않는다. 검사 기준은
과거 데이터와 M2 추정이 정한다. 위반율은 M2·M3를 고치라는 계기판이다."


======================================================================
## M5 — 검증·군집화
======================================================================

### 쉬운 설명

마지막 단계. 세 가지 일을 합니다.

(a) 원 단위 되돌리기
  M3 시나리오는 아직 '변화량' 공간(M1에서 차분한 상태). 사람이 읽을 수
  있는 실제 집값 단위로 되돌립니다. M1의 transform_log를 역으로 적용 —
  차분했던 변수는 마지막 실제값에서 시작해 변화량을 차곡차곡 더함.

(b) 성적표 매기기 (검증)
  생성된 시나리오 셋이 좋은지 4가지로 채점:
  - Coverage : 실제값이 시나리오 범위 안에 들어오나 (포괄력)
  - Width    : 그 범위가 쓸데없이 넓진 않나 (정보력)
  - Rank     : 실제값이 분포 중앙에 골고루 오나, 한쪽에 쏠리나 (확률 정확도)
  - Realism  : M4 위반율 (경로 현실성)

(c) 상승/보합/하락 군집
  수천 개 시나리오를 최종 집값 수익률로 정렬해 위·중간·아래 1/3로 나눔.
  중요: 국면을 미리 정해놓고 경로를 만든 게 아니라(기존 방식의 편향),
        자유롭게 만든 다음 사후에 분류 → 더 공정함.

### 코드

```python
import numpy as np
import pandas as pd

def run_M5(scenarios, transform_log, feature_cols, target_col,
           actual_future=None):
    """역변환 → 검증 → 상승/보합/하락 군집."""

    # --- (a) 원 단위 복원 ---
    restored = scenarios.copy()
    for col, info in transform_log.items():
        if col not in feature_cols or not info["diff"]:
            continue
        j = feature_cols.index(col)
        # 차분 역변환: 마지막 실제값 + 변화량 누적합
        restored[:, :, j] = (info["last_level"]
                             + np.cumsum(scenarios[:, :, j], axis=1))

    tgt = feature_cols.index(target_col)
    tgt_paths = restored[:, :, tgt]          # (N, horizon) 집값 경로

    # --- (b) 검증 성적표 (실제 미래값이 있을 때) ---
    report = {}
    if actual_future is not None:
        p10 = np.percentile(tgt_paths, 10, axis=0)
        p90 = np.percentile(tgt_paths, 90, axis=0)
        inside = (actual_future >= p10) & (actual_future <= p90)
        report["coverage"] = float(inside.mean())
        report["width"]    = float((p90 - p10).mean())
        # Rank: 실제값이 분포 어디쯤 — 0.5 근처가 이상적
        ranks = [(tgt_paths[:, t] < actual_future[t]).mean()
                 for t in range(len(actual_future))]
        report["rank_mean"] = float(np.mean(ranks))

    # --- (c) 상승/보합/하락 군집 ---
    final_return = tgt_paths[:, -1] / tgt_paths[:, 0] - 1
    order = np.argsort(final_return)
    n = len(order)
    groups = {
        "하락": order[:n // 3],
        "보합": order[n // 3: 2 * n // 3],
        "상승": order[2 * n // 3:],
    }
    bands = {}
    for name, idx in groups.items():
        g = tgt_paths[idx]
        bands[name] = {
            "p10": np.percentile(g, 10, axis=0),
            "p50": np.percentile(g, 50, axis=0),
            "p90": np.percentile(g, 90, axis=0),
        }

    return {"restored": restored, "target_paths": tgt_paths,
            "validation": report, "clusters": bands}
```

### 한 줄 요약
"시나리오를 실제 집값 단위로 되돌리고, Coverage·Width·Rank·Realism으로
채점한 뒤, 상승·보합·하락 세 그룹의 가격 밴드로 정리한다."


======================================================================
## 전체 실행 (파이프라인 한 번에)
======================================================================

```python
def run_pipeline(raw_df, feature_cols, policy_cols, target_col,
                 actual_future=None):
    # M1: 전처리
    stat_df, tlog = run_M1(raw_df, feature_cols, policy_cols)
    # M2: VAR 엔진
    var_result = run_M2(stat_df, feature_cols)
    # M3: 시나리오 수천 개
    scenarios = run_M3(stat_df, var_result, feature_cols)
    # M4: 현실성 검사 (리포트 모드)
    #     검사는 원단위에서 하므로 임시 복원본으로 점검
    tmp = run_M5(scenarios, tlog, feature_cols, target_col)["restored"]
    _, realism = run_M4(tmp, raw_df, var_result, feature_cols, target_col)
    # M5: 검증·군집
    result = run_M5(scenarios, tlog, feature_cols, target_col,
                    actual_future)
    result["realism"] = realism
    return result

# 사용 예
# out = run_pipeline(df, FEATURES, POLICY_COLS, "reb__apt_sale_avg_price",
#                    actual_future=actual_2024)
# print(out["validation"])   # Coverage / Width / Rank
# print(out["realism"]["violation_rate"])
# print(out["clusters"]["상승"]["p50"])
```


======================================================================
## 핵심 메시지 정리
======================================================================

| 모듈 | 한 일 | 쉬운 비유 |
|------|-------|-----------|
| M1 | 데이터 정상화(차분) | VAR이 소화할 수 있게 재료 손질 |
| M2 | VAR 관계 추정 | 변수들의 '영향 관계' 학습 = 엔진 |
| M3 | 미래로 굴리기 | 엔진을 수천 번 돌려 미래 경로 생성 |
| M3+ | N·재료 근거화 | 트라이 수를 하드조건으로 역산, 잔차주머니 보강 |
| M4 | 현실성 검사 | 가짜 경로 잡는 검사관 + 계기판 |
| M5 | 복원·검증·군집 | 채점하고 상승/보합/하락으로 정리 |

설계 전체를 관통하는 원칙:
- 사람이 직관으로 박는 숫자를 최대한 없앤다.
- 어쩔 수 없이 상수로 두는 건 '변수 정의'(금리≥0)나
  '보수성 노브'(상위 1%로 볼지)뿐 — 둘 다 경제 가정값이 아님.
- 실질 기준은 전부 과거 데이터(M1)와 추정 결과(M2)에서 나온다.
- M4는 안 버리고 리포트한다 — 확률 비중 왜곡 방지.
- 트라이 수 N은 "도박"이 아니라 하드 조건(99% 확률로 극단 최소 k번)에서
  수학으로 역산한다. 단 재료(잔차 주머니)를 먼저 채워야 공식이 작동한다.
- 위반율·검증 점수가 다음에 무엇을 고칠지 알려준다.

(2번 트랙) M5 검증이 "분포는 맞는데 인과·정책개입 질문에서 막힌다"를
가리키면, M2를 SVAR로 확장해 충격을 인과적으로 식별한다.
```

