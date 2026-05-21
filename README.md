# 고령 취약계층 헬스케어 수요 예측

공공데이터 기반 ML 파이프라인 | 서울시 행정동 단위 횡단면 분석 (2023)

> 머신러닝 강의 프로젝트 — 수정본 v4

📋 **[프로젝트 계획서 (수정본 v4)](https://htmlpreview.github.io/?https://raw.githubusercontent.com/Hufssiyoung/elderly_healthcare_Income_analysis/main/%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8_%EA%B3%84%ED%9A%8D_%EC%88%98%EC%A0%95%EB%B3%B8.html)**

---

## 프로젝트 설정

### 요구사항

- Python 3.10+
- Jupyter Notebook 또는 JupyterLab

### 설치

```bash
# 저장소 클론
git clone <repository-url>
cd HUFS_BA

# 의존성 설치
pip install -r requirements.txt
```

### 실행 순서

```bash
# JupyterLab 실행
jupyter lab
```

| 순서 | 노트북 | 설명 |
|------|--------|------|
| 1 | `TrackA_Regression.ipynb` | 요양수요가속도 회귀 예측 |
| 2 | `TrackB_Classification.ipynb` | IMD 위험등급 분류 |
| 3 | `TrackC_Clustering.ipynb` | 지역 특성별 군집화 |

> **데이터 파일**: `data/` 디렉토리에 아래 파일이 있어야 합니다.
> `df_analysis.csv`, `df_criteria.csv`, `HangJeongDong_ver20260201.geojson`

---

## 프로젝트 개요

서울시 425~430개 행정동을 분석 단위로, 2018→2023년 독거노인 수 변화율(요양수요가속도)을 예측·분류·군집화하는 ML 파이프라인을 구축합니다.

- **분석 단위**: 서울시 행정동 (~420개)
- **분석 방식**: 2023년 기준 횡단면 분석
- **Y 변수**: 요양수요가속도 = `ln(독거노인수_2023 / 독거노인수_2018) × 100`

---

## 트랙 구성

| 트랙 | 유형 | 목표 | 알고리즘 |
|------|------|------|----------|
| **Track A** | 회귀 (Supervised) | 요양수요가속도 예측 | Random Forest, XGBoost, Ridge/Lasso, Decision Tree |
| **Track B** | 분류 (Supervised) | IMD 3분위 위험등급 분류 | Logistic Regression, SVM (RBF), Random Forest, XGBoost |
| **Track C** | 군집화 (Unsupervised) | 지역 특성별 군집화 | K-Means, DBSCAN |

Track B 레이블: IMD(Index of Multiple Deprivation) 기반 **고위험·중위험·저위험** 각 140개 (균형 클래스)

---

## 데이터 소스

| 데이터 | 단위 | 용도 |
|--------|------|------|
| 주민등록 인구통계 (2022~2026) | 행정동 | 고령화율, 연령 구성 |
| 독거노인 현황 (2018, 2022~2025) | 행정동 | Y 변수 원천 |
| NHIS 진료내역 (2023) | 서울 전체 | 연령 가중치 산출 |
| 건강_병원.csv | 좌표 기반 | 행정동별 의료기관수 |
| 서울시 노인의료복지시설현황.xlsx | 자치구 | 요양시설 포화도 |
| HangJeongDong_ver20260201.geojson | 행정동 | 공간 조인 및 지도 시각화 |
| 국민건강보험공단 장기요양기관 현황 | 법정동 | Track B D2 요양시설접근성 |
| 기초생활수급자 현황 (2024) | 행정동 | Track B D1 소득박탈 |
| 지역사회건강조사 | 자치구 | Track B D3 복지기반 / D4 건강위험 |

---

## 최종 피처 (X 변수 9개 — df_analysis.csv)

| 변수 | VIF | r(Y) | 유형 |
|------|-----|------|------|
| 고령화율 | 1.95 | -0.02 | 인구통계 |
| 고령화율_변화 (2022→2023) | 1.73 | +0.24 | 트렌드 |
| 독거노인비율 | 1.38 | +0.35 | 인구통계 |
| 비율_75-79세 | 1.32 | -0.29 | 연령구성 |
| 비율_85세이상 | 1.16 | -0.21 | 연령구성 |
| 인프라갭 (65세이상인구 / (의료기관수+1)) | 1.54 | +0.01 | 인프라 |
| 포화도 (요양시설 현원/정원) | 1.65 | -0.17 | 인프라 |
| 시설수 | 2.53 | +0.16 | 인프라 |
| 수요공급갭지수 | 2.33 | -0.19 | 공급부족 |

> **VIF 계산**: statsmodels의 무절편 R² 왜곡 버그를 피해 `sklearn LinearRegression(intercept=True)` 기반으로 직접 산출

---

## 파이프라인 구조

```
1. 데이터 수집 및 병합
   ├── normalize_dong() 행정동명 정규화
   ├── 의료기관 좌표계 변환 (EPSG:5174 → 4326)
   └── Track B IMD 레이블 생성 (track2_criteria.ipynb → df_criteria.csv)

2. 탐색적 데이터 분석 (EDA)
   ├── 분포·이상치 시각화
   ├── 피처 간 상관 히트맵
   └── 서울 행정동 지도 시각화 (Moran's I)

3. 피처 엔지니어링
   ├── 공간 조인 (geopandas sjoin)
   └── VIF 스크리닝 (Step 0~5) → feat_cols 확정 → KNN Imputer

4. 모델링 (Track A / B / C)

5. 모델 평가 및 비교
   ├── 회귀: RMSE / MAE / R² / MAPE
   ├── 분류: Macro F1 / Precision / Recall / Confusion Matrix
   └── 군집화: Silhouette Score / Davies-Bouldin / Elbow Curve

6. 결과 해석
   ├── SHAP 피처 중요도
   └── 고위험 행정동 choropleth 지도
```

---

## Track B IMD 구성

| 도메인 | 가중치 | 지표 | 단위 |
|--------|--------|------|------|
| D1 소득박탈 | 30% | 기초생활수급자수 / 65세이상인구 | 행정동 |
| D2 요양시설접근성 | 25% | NHIS 장기요양기관수 / 65세이상인구 × 1000 | 법정동→행정동 |
| D3 복지기반 | 25% | 지역사회건강조사 복지 취약 지표 | 자치구→행정동 |
| D4 건강위험 | 20% | 지역사회건강조사 건강행태 지표 | 자치구→행정동 |

도메인 간 상관계수 최대 0.238 (독립성 확인 ✓) | 처리 순서: Z-score → exp 변환 → 가중합 → 3분위 분류

---

## 데이터 주의사항

- **행정동명 불일치**: `normalize_dong()`으로 표기 통일 (창신제1동→창신1동, 정능→정릉 등)
- **신사동 중복** (관악구·강남구): 행정동코드 기준 하드코딩 처리
- **Track B 행정동 불일치 6개**: df_analysis(426개) vs df_criteria(420개) — 강남구 개포3동, 강동구 강일동·상일제1·2동·둔촌제1동, 구로구 항동 → 신설·개편 행정동으로 매핑 불가, 학습 시 제외 또는 자치구 평균 대체
- **결측치 처리**: 의료기관수 NaN → 0 / 나머지 9개 피처 → KNN Imputer (n_neighbors=5)

---

## 기술 스택

```
Python 3.10+
pandas / numpy           데이터 처리
scikit-learn             ML 전반 + VIF 산출
xgboost / lightgbm       Gradient Boosting
shap                     피처 중요도 해석
geopandas / folium       공간 분석 및 지도 시각화
matplotlib / seaborn     EDA 시각화
openpyxl                 Excel 처리
scipy                    Z-score · IMD 산출
Jupyter Notebook / Google Colab
```
