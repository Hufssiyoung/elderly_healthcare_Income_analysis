import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os as _os
import joblib as _jl

_PKL_A = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "precomputed", "track_a.pkl")
from sklearn.model_selection import (
    KFold, StratifiedKFold, cross_validate, learning_curve, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from .utils import setup_font, load_main, FEAT_COLS, TARGET

RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-9, y_true))) * 100


def build_pipe(model, scale=True):
    steps = [("scaler", StandardScaler())] if scale else []
    steps.append(("model", model))
    return Pipeline(steps)


# ── Best params from GridSearchCV (notebooks) ─────────────────────────────
BEST_MODELS = {
    "Ridge": build_pipe(Ridge(alpha=10), scale=True),
    "Lasso": build_pipe(Lasso(alpha=0.1, max_iter=10000), scale=True),
    "DecisionTree": build_pipe(
        DecisionTreeRegressor(max_depth=5, min_samples_leaf=5,
                              min_samples_split=20, random_state=RANDOM_STATE), scale=False),
    "RandomForest": build_pipe(
        RandomForestRegressor(n_estimators=200, max_depth=None, max_features="log2",
                              min_samples_leaf=2, min_samples_split=2,
                              random_state=RANDOM_STATE, n_jobs=-1), scale=False),
    "XGBoost": build_pipe(
        XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.1,
                     subsample=0.8, colsample_bytree=1.0,
                     reg_alpha=0.1, reg_lambda=1.0,
                     random_state=RANDOM_STATE, verbosity=0, n_jobs=-1), scale=False),
}
MODEL_NAMES = list(BEST_MODELS.keys())


@st.cache_resource
def prepare_track_a():
    if _os.path.exists(_PKL_A):
        return _jl.load(_PKL_A)
    df = load_main()
    X = df[FEAT_COLS].copy()
    y = df[TARGET].copy()
    y_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y_bins)
    for part in [X_tr, X_te, y_tr, y_te]:
        part.reset_index(drop=True, inplace=True)

    # KFold CV (default params)
    kf = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    y_bins_tr = pd.qcut(y_tr, q=5, labels=False, duplicates="drop").reset_index(drop=True)
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    from sklearn.base import clone
    from sklearn.metrics import make_scorer
    rmse_scorer = make_scorer(lambda yt, yp: -np.sqrt(mean_squared_error(yt, yp)),
                               greater_is_better=True)
    scoring = {"RMSE": rmse_scorer, "R2": "r2"}

    default_models = {
        "Ridge": build_pipe(Ridge(alpha=1.0), scale=True),
        "Lasso": build_pipe(Lasso(alpha=0.1, max_iter=10000), scale=True),
        "DecisionTree": build_pipe(DecisionTreeRegressor(random_state=RANDOM_STATE), scale=False),
        "RandomForest": build_pipe(RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), scale=False),
        "XGBoost": build_pipe(XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1), scale=False),
    }

    # Single holdout (default params) — 과적합 확인용
    single_val = []
    for name, pipe in default_models.items():
        m = clone(pipe)
        m.fit(X_tr, y_tr)
        p_tr_s = m.predict(X_tr)
        p_te_s = m.predict(X_te)
        single_val.append({
            "모델": name,
            "Train_RMSE": rmse(y_tr, p_tr_s),
            "Test_RMSE":  rmse(y_te, p_te_s),
            "Train_R²":   r2_score(y_tr, p_tr_s),
            "Test_R²":    r2_score(y_te, p_te_s),
            "Overfit_Gap": rmse(y_te, p_te_s) - rmse(y_tr, p_tr_s),
        })
    df_single = pd.DataFrame(single_val)

    cv_kf, cv_skf = {}, {}
    for name, pipe in default_models.items():
        cv_kf[name] = cross_validate(pipe, X_tr, y_tr, cv=kf, scoring=scoring,
                                     return_train_score=True, n_jobs=-1)
        # StratKFold manual
        res = {"train_RMSE": [], "test_RMSE": [], "train_R2": [], "test_R2": []}
        for tr_idx, va_idx in skf.split(X_tr, y_bins_tr):
            m = clone(pipe)
            m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
            p_tr = m.predict(X_tr.iloc[tr_idx])
            p_va = m.predict(X_tr.iloc[va_idx])
            res["train_RMSE"].append(rmse(y_tr.iloc[tr_idx], p_tr))
            res["test_RMSE"].append(rmse(y_tr.iloc[va_idx], p_va))
            res["train_R2"].append(r2_score(y_tr.iloc[tr_idx], p_tr))
            res["test_R2"].append(r2_score(y_tr.iloc[va_idx], p_va))
        cv_skf[name] = {k: np.array(v) for k, v in res.items()}

    # Tuned model — train once
    from sklearn.base import clone as _clone
    tuned = {}
    test_res = []
    for name, pipe in BEST_MODELS.items():
        m = _clone(pipe)
        m.fit(X_tr, y_tr)
        p_te = m.predict(X_te)
        p_tr2 = m.predict(X_tr)
        tuned[name] = m
        test_res.append({
            "모델": name,
            "Train_RMSE": rmse(y_tr, p_tr2), "Test_RMSE": rmse(y_te, p_te),
            "Train_R2": r2_score(y_tr, p_tr2), "Test_R2": r2_score(y_te, p_te),
            "Test_MAE": mean_absolute_error(y_te, p_te),
            "Overfit_Gap": rmse(y_te, p_te) - rmse(y_tr, p_tr2),
        })
    df_test = pd.DataFrame(test_res).sort_values("Test_RMSE")

    best_name = df_test.iloc[0]["모델"]
    best_model = tuned[best_name]
    y_pred = best_model.predict(X_te)
    residuals = y_te.values - y_pred

    return {
        "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
        "df_single": df_single,
        "cv_kf": cv_kf, "cv_skf": cv_skf,
        "tuned": tuned, "df_test": df_test,
        "best_name": best_name, "best_model": best_model,
        "y_pred": y_pred, "residuals": residuals,
    }


def show():
    setup_font()
    st.title("📈 Track A — 요양수요가속도 회귀분석")
    st.markdown("**모델**: Ridge · Lasso · DecisionTree · RandomForest · XGBoost")

    with st.spinner("모델 학습 중... (최초 1회만 실행)"):
        d = prepare_track_a()

    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📌 단일 검증", "🔁 교차검증", "📉 과적합 진단", "📐 학습곡선", "🔬 잔차분석", "🧠 SHAP"]
    )

    # ── Tab 0: 단일 검증 ───────────────────────────────────────────────────
    with tab0:
        st.subheader("단일 검증 (Hold-out) — 기본 하이퍼파라미터")
        st.markdown(
            "교차검증 도입 이전, **단순 Train/Test 분할(80/20)** 로만 평가한 결과입니다. "
            "기본(default) 하이퍼파라미터를 사용했으며, 트리 계열 모델에서 심각한 과적합이 관찰됩니다."
        )

        df_s = d["df_single"]
        DEFAULT_MODEL_NAMES = df_s["모델"].tolist()

        # 수치 테이블
        st.dataframe(
            df_s.set_index("모델").style.format("{:.3f}").background_gradient(
                subset=["Overfit_Gap"], cmap="RdYlGn_r"
            ),
            width='stretch',
        )

        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        x = np.arange(len(DEFAULT_MODEL_NAMES))
        w = 0.38

        # RMSE 비교
        axes[0].bar(x - w/2, df_s["Train_RMSE"], w, label="Train", color="steelblue", alpha=0.85, edgecolor="white")
        axes[0].bar(x + w/2, df_s["Test_RMSE"],  w, label="Test",  color="tomato",    alpha=0.85, edgecolor="white")
        axes[0].set_xticks(x); axes[0].set_xticklabels(DEFAULT_MODEL_NAMES, rotation=20, ha="right")
        axes[0].set_title("RMSE 비교 (Train vs Test)", fontsize=11, fontweight="bold")
        axes[0].legend(fontsize=9)

        # R² 비교
        axes[1].bar(x - w/2, df_s["Train_R²"], w, label="Train", color="steelblue", alpha=0.85, edgecolor="white")
        axes[1].bar(x + w/2, df_s["Test_R²"],  w, label="Test",  color="tomato",    alpha=0.85, edgecolor="white")
        axes[1].set_xticks(x); axes[1].set_xticklabels(DEFAULT_MODEL_NAMES, rotation=20, ha="right")
        axes[1].axhline(0, color="orange", lw=1, ls="--")
        axes[1].set_title("R² 비교 (Train vs Test)", fontsize=11, fontweight="bold")
        axes[1].legend(fontsize=9)

        # Overfit Gap
        gap_colors = ["tomato" if v > 3 else "seagreen" for v in df_s["Overfit_Gap"]]
        axes[2].bar(DEFAULT_MODEL_NAMES, df_s["Overfit_Gap"], color=gap_colors, alpha=0.85, edgecolor="white")
        axes[2].axhline(0, color="gray", lw=1, ls="--")
        for i, v in enumerate(df_s["Overfit_Gap"]):
            axes[2].text(i, v + 0.05, f"{v:+.2f}", ha="center", fontsize=9)
        axes[2].set_xticklabels(DEFAULT_MODEL_NAMES, rotation=20, ha="right")
        axes[2].set_title("Overfit Gap (Test − Train RMSE)", fontsize=11, fontweight="bold")

        plt.suptitle("단일 검증 결과 — 기본 하이퍼파라미터 (Hold-out 80/20)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.warning(
            "DecisionTree는 Train RMSE ≈ 0에 가까운 완전 과적합, "
            "RandomForest·XGBoost도 Train/Test 격차가 큽니다. "
            "단일 분할은 분할 방식에 따라 결과가 크게 달라지므로 "
            "**교차검증(Cross-Validation)** 으로 전환하여 안정적인 성능을 측정했습니다."
        )

    # ── Tab 1: CV 비교 ─────────────────────────────────────────────────────
    with tab1:
        st.subheader("KFold-5 vs StratifiedKFold-5 교차검증 비교")

        rows = []
        for name in MODEL_NAMES:
            kf_rmse = -d["cv_kf"][name]["test_RMSE"].mean()
            kf_r2 = d["cv_kf"][name]["test_R2"].mean()
            sk_rmse = d["cv_skf"][name]["test_RMSE"].mean()
            sk_r2 = d["cv_skf"][name]["test_R2"].mean()
            rows.append({"모델": name,
                         "KFold Val RMSE": round(kf_rmse, 3), "KFold Val R²": round(kf_r2, 3),
                         "StratKFold Val RMSE": round(sk_rmse, 3), "StratKFold Val R²": round(sk_r2, 3)})
        st.dataframe(pd.DataFrame(rows).set_index("모델"), width='stretch')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(MODEL_NAMES))
        w = 0.35
        for ax, (kf_key, sk_key, title) in zip(axes, [
            ("test_RMSE", "test_RMSE", "Val RMSE"),
            ("test_R2", "test_R2", "Val R²"),
        ]):
            kf_vals = [-d["cv_kf"][n][kf_key].mean() if "RMSE" in kf_key else d["cv_kf"][n][kf_key].mean()
                       for n in MODEL_NAMES]
            sk_vals = [d["cv_skf"][n][sk_key].mean() for n in MODEL_NAMES]
            ax.bar(x - w/2, kf_vals, w, label="KFold-5", color="steelblue", alpha=0.85, edgecolor="white")
            ax.bar(x + w/2, sk_vals, w, label="StratKFold-5", color="salmon", alpha=0.85, edgecolor="white")
            ax.set_xticks(x)
            ax.set_xticklabels(MODEL_NAMES, rotation=20, ha="right")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            if "R²" in title:
                ax.axhline(0, color="orange", lw=1, ls="--")
        plt.suptitle("CV 전략별 검증 성능 비교 (기본 하이퍼파라미터)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 2: 과적합 진단 ─────────────────────────────────────────────────
    with tab2:
        st.subheader("과적합 진단 — Train vs Validation RMSE 격차")
        st.markdown("GridSearchCV 튜닝 후 테스트셋 최종 결과")

        df_test = d["df_test"]
        st.dataframe(
            df_test[["모델", "Train_RMSE", "Test_RMSE", "Train_R2", "Test_R2", "Test_MAE", "Overfit_Gap"]]
            .set_index("모델").style.format("{:.3f}").background_gradient(
                subset=["Overfit_Gap"], cmap="RdYlGn_r"),
            width='stretch',
        )

        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        sorted_names = df_test["모델"].tolist()
        x = np.arange(len(sorted_names))
        w = 0.38

        for ax, (col_tr, col_te, title) in zip(axes, [
            ("Train_RMSE", "Test_RMSE", "RMSE 비교"),
            ("Train_R2", "Test_R2", "R² 비교"),
            ("Overfit_Gap", None, "과적합 갭 (Test - Train RMSE)"),
        ]):
            if col_te is None:
                vals = [df_test[df_test["모델"] == n][col_tr].values[0] for n in sorted_names]
                colors = ["tomato" if v > 3 else "seagreen" for v in vals]
                ax.bar(sorted_names, vals, color=colors, alpha=0.85, edgecolor="white")
                ax.axhline(0, color="gray", lw=1, ls="--")
                for i, v in enumerate(vals):
                    ax.text(i, v + 0.05, f"{v:+.2f}", ha="center", fontsize=9)
            else:
                tr_v = [df_test[df_test["모델"] == n][col_tr].values[0] for n in sorted_names]
                te_v = [df_test[df_test["모델"] == n][col_te].values[0] for n in sorted_names]
                ax.bar(x - w/2, tr_v, w, label="Train", color="steelblue", alpha=0.85, edgecolor="white")
                ax.bar(x + w/2, te_v, w, label="Test", color="tomato", alpha=0.85, edgecolor="white")
                ax.set_xticks(x)
                ax.legend(fontsize=8)
                if "R²" in title:
                    ax.axhline(0, color="orange", lw=1, ls="--")
            ax.set_xticklabels(sorted_names, rotation=20, ha="right")
            ax.set_title(title, fontsize=11, fontweight="bold")
        plt.suptitle("최종 모델 성능 비교 (튜닝 후 Holdout Test)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 3: 학습 곡선 ───────────────────────────────────────────────────
    with tab3:
        st.subheader("학습 곡선 — 훈련 샘플 수에 따른 Train/Val RMSE")
        st.info("최초 실행 시 약 30초 소요됩니다.")

        @st.cache_resource
        def compute_learning_curves(_X_tr, _y_tr):
            from sklearn.model_selection import learning_curve as lc
            from sklearn.metrics import make_scorer
            rmse_sc = make_scorer(lambda yt, yp: -np.sqrt(mean_squared_error(yt, yp)),
                                   greater_is_better=True)
            kf = KFold(5, shuffle=True, random_state=RANDOM_STATE)
            results = {}
            for name, pipe in BEST_MODELS.items():
                from sklearn.base import clone
                ts, tr_sc, va_sc = lc(
                    clone(pipe), _X_tr, _y_tr,
                    train_sizes=np.linspace(0.15, 1.0, 8),
                    cv=kf, scoring=rmse_sc, n_jobs=-1,
                )
                results[name] = (ts, -tr_sc, -va_sc)
            return results

        lc_results = d.get("learning_curves") or compute_learning_curves(d["X_tr"], d["y_tr"])

        fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=False)
        for ax, name in zip(axes, MODEL_NAMES):
            ts, tr_sc, va_sc = lc_results[name]
            tr_m, tr_s = tr_sc.mean(1), tr_sc.std(1)
            va_m, va_s = va_sc.mean(1), va_sc.std(1)
            ax.plot(ts, tr_m, "o-", color="steelblue", lw=2, label="Train RMSE")
            ax.fill_between(ts, tr_m - tr_s, tr_m + tr_s, alpha=0.2, color="steelblue")
            ax.plot(ts, va_m, "s--", color="tomato", lw=2, label="Val RMSE")
            ax.fill_between(ts, va_m - va_s, va_m + va_s, alpha=0.2, color="tomato")
            ax.set_title(name, fontsize=10, fontweight="bold")
            ax.set_xlabel("훈련 샘플 수")
            ax.set_ylabel("RMSE")
            ax.legend(fontsize=7)
        plt.suptitle("학습 곡선 — 훈련 샘플 수에 따른 Train/Val RMSE", fontsize=13, fontweight="bold", y=1.04)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 4: 잔차 분석 ───────────────────────────────────────────────────
    with tab4:
        st.subheader(f"잔차 분석 — 최적 모델: {d['best_name']}")

        residuals = d["residuals"]
        y_pred = d["y_pred"]
        y_te = d["y_te"]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 잔차 분포
        axes[0].hist(residuals, bins=25, color="purple", edgecolor="white", alpha=0.85)
        axes[0].axvline(0, color="orange", lw=2, ls="--")
        axes[0].axvline(residuals.mean(), color="pink", lw=1.5, ls="--",
                        label=f"평균={residuals.mean():.2f}")
        axes[0].set_title("잔차 분포", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("잔차 (실제 - 예측)")
        axes[0].legend()

        # 실제 vs 예측
        mn, mx = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
        sc = axes[1].scatter(y_te, y_pred, alpha=0.6, s=30,
                             c=np.abs(residuals), cmap="RdYlGn_r", edgecolors="none")
        axes[1].plot([mn, mx], [mn, mx], "k--", lw=1.5, label="완벽한 예측")
        plt.colorbar(sc, ax=axes[1], shrink=0.8, label="|잔차|")
        axes[1].set_xlabel("실제 요양수요가속도")
        axes[1].set_ylabel("예측 요양수요가속도")
        axes[1].set_title("실제 vs 예측", fontsize=12, fontweight="bold")
        axes[1].legend()

        # 이분산성
        axes[2].scatter(y_pred, residuals, alpha=0.6, s=30,
                        c=np.abs(residuals), cmap="RdYlGn_r", edgecolors="none")
        axes[2].axhline(0, color="orange", lw=2, ls="--")
        axes[2].set_xlabel("예측값")
        axes[2].set_ylabel("잔차")
        axes[2].set_title("잔차 vs 예측값 (이분산성)", fontsize=12, fontweight="bold")

        plt.suptitle(f"잔차 분석 — {d['best_name']}", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        col1, col2, col3 = st.columns(3)
        col1.metric("잔차 평균", f"{residuals.mean():.3f}")
        col2.metric("잔차 표준편차", f"{residuals.std():.3f}")
        col3.metric("Test R²", f"{r2_score(y_te, y_pred):.3f}")

    # ── Tab 5: SHAP ────────────────────────────────────────────────────────
    with tab5:
        st.subheader(f"SHAP 피처 중요도 — {d['best_name']}")

        @st.cache_resource
        def compute_shap(_model, _X_te, _X_tr, feat_cols):
            import shap
            if hasattr(_model, "named_steps"):
                core = _model.named_steps[list(_model.named_steps.keys())[-1]]
                if "scaler" in _model.named_steps:
                    X_tr_t = pd.DataFrame(_model.named_steps["scaler"].transform(_X_tr), columns=feat_cols)
                    X_te_t = pd.DataFrame(_model.named_steps["scaler"].transform(_X_te), columns=feat_cols)
                else:
                    X_tr_t, X_te_t = _X_tr.copy(), _X_te.copy()
            else:
                core, X_tr_t, X_te_t = _model, _X_tr.copy(), _X_te.copy()
            try:
                explainer = shap.TreeExplainer(core)
                sv = explainer.shap_values(X_te_t)
            except Exception:
                explainer = shap.LinearExplainer(core, X_tr_t)
                sv = explainer.shap_values(X_te_t)
            return sv, X_te_t

        if "shap_a" in d:
            shap_vals, X_te_t = d["shap_a"]
        else:
            shap_vals, X_te_t = compute_shap(
                d["best_model"], d["X_te"], d["X_tr"], FEAT_COLS)

        import shap as shap_lib
        mean_abs = np.abs(shap_vals).mean(axis=0)
        shap_df = pd.DataFrame({"피처": FEAT_COLS, "Mean |SHAP|": mean_abs}).sort_values("Mean |SHAP|")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 전역 중요도 막대
        axes[0].barh(shap_df["피처"], shap_df["Mean |SHAP|"],
                     color=plt.cm.RdBu_r(np.linspace(0.2, 0.8, len(FEAT_COLS))),
                     edgecolor="white", alpha=0.85)
        axes[0].set_title("SHAP 전역 중요도 (Mean |SHAP|)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Mean |SHAP value|")

        # Beeswarm
        plt.sca(axes[1])
        shap_lib.summary_plot(shap_vals, X_te_t, feature_names=FEAT_COLS,
                              show=False, plot_type="dot", max_display=10)
        axes[1].set_title("SHAP Beeswarm — 피처 영향 방향", fontsize=12, fontweight="bold")

        plt.suptitle(f"SHAP 피처 중요도 ({d['best_name']})", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(shap_df.sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True),
                     width='stretch')
