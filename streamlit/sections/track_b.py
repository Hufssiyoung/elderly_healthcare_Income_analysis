import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os as _os
import joblib as _jl

_PKL_B = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "precomputed", "track_b.pkl")
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, learning_curve, train_test_split, GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, make_scorer, precision_score, recall_score
)
from xgboost import XGBClassifier
from .utils import setup_font, load_merged, FEAT_COLS, CLASS_ORDER

RANDOM_STATE = 42
class_map = {c: i for i, c in enumerate(CLASS_ORDER)}
inv_map = {i: c for c, i in class_map.items()}


def build_pipe(model, scale=True):
    steps = [("scaler", StandardScaler())] if scale else []
    steps.append(("model", model))
    return Pipeline(steps)


TUNED_MODELS = {
    "LogisticRegression": build_pipe(
        LogisticRegression(C=1.0, l1_ratio=0, max_iter=2000, random_state=RANDOM_STATE), scale=True),
    "SVM_RBF": build_pipe(
        SVC(kernel="rbf", C=50, gamma="scale", probability=True, random_state=RANDOM_STATE), scale=True),
    "RandomForest": build_pipe(
        RandomForestClassifier(n_estimators=300, max_depth=7, min_samples_leaf=5,
                               max_features=0.5, random_state=RANDOM_STATE, n_jobs=-1), scale=False),
    "XGBoost": build_pipe(
        XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                      subsample=1.0, colsample_bytree=0.8,
                      reg_alpha=0.1, reg_lambda=2.0,
                      random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
                      eval_metric="mlogloss"), scale=False),
}
MODEL_NAMES = list(TUNED_MODELS.keys())
sorted_labels = sorted(CLASS_ORDER)  # 알파벳 순
xgb_col_order = [class_map[c] for c in sorted_labels]


def _decode_xgb(preds_int):
    return pd.Categorical([inv_map[i] for i in preds_int], categories=CLASS_ORDER, ordered=True)


@st.cache_resource
def prepare_track_b():
    if _os.path.exists(_PKL_B):
        return _jl.load(_PKL_B)
    from sklearn.base import clone
    df = load_merged()
    X = df[FEAT_COLS].copy()
    y = df["위험등급"].copy()
    y_enc = y.map(class_map)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    for part in [X_tr, X_te, y_tr, y_te]:
        part.reset_index(drop=True, inplace=True)
    y_tr_enc = y_tr.map(class_map)
    y_te_enc = y_te.map(class_map)

    # SKFold CV (default params)
    cv_skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    f1_scorer = "f1_macro"
    auc_scorer = make_scorer(roc_auc_score, response_method="predict_proba",
                             multi_class="ovr", average="macro")
    scoring = {"F1": f1_scorer, "Accuracy": "accuracy",
               "Precision": "precision_macro", "Recall": "recall_macro",
               "AUC": auc_scorer}

    cv_results = {}
    for name, pipe in TUNED_MODELS.items():
        y_cv = y_tr_enc if name == "XGBoost" else y_tr
        cv_results[name] = cross_validate(
            clone(pipe), X_tr, y_cv, cv=cv_skf, scoring=scoring,
            return_train_score=True, n_jobs=-1)

    # Tuned models — train once
    trained, test_preds, test_probs = {}, {}, {}
    test_results = []
    for name, pipe in TUNED_MODELS.items():
        m = clone(pipe)
        y_fit = y_tr_enc if name == "XGBoost" else y_tr
        m.fit(X_tr, y_fit)
        trained[name] = m

        p_te = m.predict(X_te)
        prob_te = m.predict_proba(X_te)
        if name == "XGBoost":
            p_te = _decode_xgb(p_te)
            prob_te = prob_te[:, xgb_col_order]

        test_preds[name] = p_te
        test_probs[name] = prob_te
        test_results.append({
            "모델": name,
            "Train_F1": f1_score(y_tr, _decode_xgb(m.predict(X_tr)) if name == "XGBoost"
                                 else m.predict(X_tr), average="macro"),
            "Test_F1": f1_score(y_te, p_te, average="macro"),
            "Test_Acc": accuracy_score(y_te, p_te),
            "Test_AUC": roc_auc_score(y_te, prob_te, multi_class="ovr", average="macro"),
        })

    df_test = pd.DataFrame(test_results).sort_values("Test_F1", ascending=False)
    best_name = df_test.iloc[0]["모델"]

    return {
        "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
        "y_tr_enc": y_tr_enc, "y_te_enc": y_te_enc,
        "cv_results": cv_results, "trained": trained,
        "test_preds": test_preds, "test_probs": test_probs,
        "df_test": df_test, "best_name": best_name,
    }


def show():
    setup_font()
    st.title("🎯 Track B — 고위험 지역 분류 (IMD 위험등급)")
    st.markdown("**모델**: LogisticRegression · SVM(RBF) · RandomForest · XGBoost")

    with st.spinner("모델 학습 중... (최초 1회만 실행)"):
        d = prepare_track_b()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📦 피처 탐색", "🔁 교차검증", "⚙️ 하이퍼파라미터",
        "📊 혼동행렬", "📐 학습곡선", "🧠 SHAP"
    ])

    df_all = load_merged()

    # ── Tab 1: 피처 탐색 ───────────────────────────────────────────────────
    with tab1:
        st.subheader("피처별 위험등급 간 분포 비교 (Box Plot)")
        colors_cls = {"저위험": "steelblue", "중위험": "goldenrod", "고위험": "tomato"}

        fig, axes = plt.subplots(2, 5, figsize=(22, 9))
        axes = axes.flatten()
        for i, feat in enumerate(FEAT_COLS):
            data_by = [df_all[df_all["위험등급"] == c][feat].values for c in CLASS_ORDER]
            bp = axes[i].boxplot(data_by, patch_artist=True,
                                 medianprops=dict(color="white", linewidth=2))
            for patch, c in zip(bp["boxes"], CLASS_ORDER):
                patch.set_facecolor(colors_cls[c])
                patch.set_alpha(0.75)
            axes[i].set_xticklabels(CLASS_ORDER, fontsize=9)
            axes[i].set_title(feat, fontsize=10, fontweight="bold")
            axes[i].grid(axis="y", alpha=0.3)
        plt.suptitle("피처별 위험등급 분포 비교", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.subheader("피처 vs 위험등급 상관계수")
        y_num = df_all["위험등급"].cat.codes
        corr = df_all[FEAT_COLS].corrwith(y_num).sort_values(key=abs, ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        bar_colors = ["tomato" if v > 0 else "steelblue" for v in corr.values]
        ax.barh(corr.index, corr.values, color=bar_colors, edgecolor="white", alpha=0.85)
        ax.axvline(0, color="white", lw=1)
        ax.set_title("피처 vs 위험등급 상관계수 (저위험=0, 중위험=1, 고위험=2)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Pearson r")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 2: 교차검증 ────────────────────────────────────────────────────
    with tab2:
        st.subheader("StratifiedKFold-5 교차검증 결과")
        rows = []
        for name in MODEL_NAMES:
            res = d["cv_results"][name]
            rows.append({
                "모델": name,
                "Val F1": round(res["test_F1"].mean(), 4),
                "Val F1 std": round(res["test_F1"].std(), 4),
                "Val Acc": round(res["test_Accuracy"].mean(), 4),
                "Val AUC": round(res["test_AUC"].mean(), 4),
                "과적합 F1 갭": round(res["train_F1"].mean() - res["test_F1"].mean(), 4),
            })
        st.dataframe(pd.DataFrame(rows).set_index("모델").style.background_gradient(
            subset=["과적합 F1 갭"], cmap="RdYlGn_r"), width='stretch')

        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        for ax, (col, title) in zip(axes, [
            ("Val F1", "Val Macro F1"),
            ("Val AUC", "Val ROC-AUC (OvR)"),
            ("과적합 F1 갭", "과적합 갭 (Train - Val F1)"),
        ]):
            vals = [rows[i][col] for i in range(len(MODEL_NAMES))]
            colors = ["tomato" if (col == "과적합 F1 갭" and v > 0.1) else "steelblue" for v in vals]
            ax.bar(MODEL_NAMES, vals, color=colors, alpha=0.85, edgecolor="white")
            ax.set_xticklabels(MODEL_NAMES, rotation=15, ha="right", fontsize=9)
            ax.set_title(title, fontsize=11, fontweight="bold")
            if col == "과적합 F1 갭":
                ax.axhline(0, color="white", lw=1, ls="--")
            for i, v in enumerate(vals):
                ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
        plt.suptitle("StratifiedKFold-5 CV 성능 비교", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 3: 하이퍼파라미터 ─────────────────────────────────────────────
    with tab3:
        st.subheader("하이퍼파라미터 탐색 결과")
        st.markdown("GridSearchCV 결과를 요약한 시각화입니다. (노트북에서 탐색, Streamlit에서 재현)")

        col_lr, col_svm = st.columns(2)

        with col_lr:
            st.markdown("**Logistic Regression — C 탐색**")
            c_range = [0.001, 0.01, 0.1, 1, 10, 100]

            @st.cache_data
            def lr_c_search(_X_tr, _y_tr):
                skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
                scores = []
                from sklearn.base import clone
                for c in c_range:
                    pipe = build_pipe(LogisticRegression(C=c, max_iter=2000,
                                                         random_state=RANDOM_STATE), scale=True)
                    res = cross_validate(clone(pipe), _X_tr, _y_tr, cv=skf,
                                         scoring="f1_macro", return_train_score=False)
                    scores.append(res["test_score"].mean())
                return scores

            lr_scores = d.get("lr_c_scores") or lr_c_search(d["X_tr"], d["y_tr"])
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.semilogx(c_range, lr_scores, "o-", color="steelblue", lw=2)
            best_c = c_range[np.argmax(lr_scores)]
            ax.axvline(best_c, color="tomato", ls="--", lw=1.5, label=f"최적 C={best_c}")
            ax.set_xlabel("C (log scale)")
            ax.set_ylabel("CV Macro F1")
            ax.set_title("LR C 탐색", fontsize=11)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_svm:
            st.markdown("**SVM RBF — C × gamma CV F1 히트맵**")

            @st.cache_data
            def svm_grid_search(_X_tr, _y_tr):
                skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
                c_vals = [0.1, 1, 10, 50, 100]
                g_vals = ["scale", "auto", 0.01, 0.1]
                grid = np.zeros((len(c_vals), len(g_vals)))
                from sklearn.base import clone
                for i, c in enumerate(c_vals):
                    for j, g in enumerate(g_vals):
                        pipe = build_pipe(SVC(kernel="rbf", C=c, gamma=g,
                                              probability=True, random_state=RANDOM_STATE), scale=True)
                        r = cross_validate(clone(pipe), _X_tr, _y_tr, cv=skf,
                                            scoring="f1_macro", return_train_score=False)
                        grid[i, j] = r["test_score"].mean()
                return grid, c_vals, [str(g) for g in g_vals]

            if "svm_grid" in d:
                grid, c_vals, g_labels = d["svm_grid"]
            else:
                grid, c_vals, g_labels = svm_grid_search(d["X_tr"], d["y_tr"])
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.heatmap(grid, annot=True, fmt=".3f", cmap="RdYlGn",
                        xticklabels=g_labels, yticklabels=c_vals, ax=ax, linewidths=0.5)
            ax.set_xlabel("gamma")
            ax.set_ylabel("C")
            ax.set_title("SVM RBF C×gamma CV Macro F1", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Tab 4: 혼동 행렬 ───────────────────────────────────────────────────
    with tab4:
        st.subheader("3×3 Confusion Matrix (4개 모델 비교)")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for ax, name in zip(axes.flatten(), MODEL_NAMES):
            cm = confusion_matrix(d["y_te"], d["test_preds"][name], labels=CLASS_ORDER)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                        xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
                        ax=ax, linewidths=0.5, vmin=0, vmax=1,
                        annot_kws={"size": 13, "weight": "bold"})
            f1 = d["df_test"][d["df_test"]["모델"] == name]["Test_F1"].values[0]
            ax.set_title(f"{name}\nMacro F1={f1:.3f}", fontsize=11, fontweight="bold")
            ax.set_xlabel("예측 클래스")
            ax.set_ylabel("실제 클래스")
        plt.suptitle("Confusion Matrix (행: 실제 / 열: 예측 | 색: 행별 비율 | 숫자: 건수)",
                     fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.subheader(f"최적 모델({d['best_name']}) — Classification Report")
        report = classification_report(d["y_te"], d["test_preds"][d["best_name"]],
                                       target_names=CLASS_ORDER, digits=4)
        st.code(report)

    # ── Tab 5: 학습 곡선 ───────────────────────────────────────────────────
    with tab5:
        st.subheader("학습 곡선 (4개 모델)")
        st.info("최초 실행 시 약 20초 소요됩니다.")

        @st.cache_resource
        def compute_lc_b(_X_tr, _y_tr, _y_tr_enc):
            from sklearn.base import clone
            skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
            results = {}
            for name, pipe in TUNED_MODELS.items():
                y_lc = _y_tr_enc if name == "XGBoost" else _y_tr
                ts, tr_sc, va_sc = learning_curve(
                    clone(pipe), _X_tr, y_lc,
                    train_sizes=np.linspace(0.15, 1.0, 8),
                    cv=skf, scoring="f1_macro", n_jobs=-1,
                )
                results[name] = (ts, tr_sc, va_sc)
            return results

        lc = d.get("lc_b") or compute_lc_b(d["X_tr"], d["y_tr"], d["y_tr_enc"])
        fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))
        for ax, name in zip(axes, MODEL_NAMES):
            ts, tr_sc, va_sc = lc[name]
            tr_m, va_m = tr_sc.mean(1), va_sc.mean(1)
            tr_s, va_s = tr_sc.std(1), va_sc.std(1)
            ax.plot(ts, tr_m, "o-", color="steelblue", lw=2, label="Train F1")
            ax.fill_between(ts, tr_m - tr_s, tr_m + tr_s, alpha=0.2, color="steelblue")
            ax.plot(ts, va_m, "s--", color="tomato", lw=2, label="Val F1")
            ax.fill_between(ts, va_m - va_s, va_m + va_s, alpha=0.2, color="tomato")
            ax.set_title(name, fontsize=10, fontweight="bold")
            ax.set_xlabel("훈련 샘플 수")
            ax.set_ylabel("Macro F1")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
        plt.suptitle("학습 곡선 — Train/Val Macro F1", fontsize=13, fontweight="bold", y=1.04)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 6: SHAP ────────────────────────────────────────────────────────
    with tab6:
        st.subheader(f"SHAP 피처 중요도 — {d['best_name']}")

        @st.cache_resource
        def compute_shap_b(_model, _X_te, _X_tr, feat_cols, name):
            import shap
            if hasattr(_model, "named_steps"):
                core = _model.named_steps[list(_model.named_steps.keys())[-1]]
                if "scaler" in _model.named_steps:
                    X_te_t = pd.DataFrame(_model.named_steps["scaler"].transform(_X_te), columns=feat_cols)
                    X_tr_t = pd.DataFrame(_model.named_steps["scaler"].transform(_X_tr), columns=feat_cols)
                else:
                    X_te_t, X_tr_t = _X_te.copy(), _X_tr.copy()
            else:
                core, X_te_t, X_tr_t = _model, _X_te.copy(), _X_tr.copy()
            try:
                explainer = shap.TreeExplainer(core)
                sv = explainer.shap_values(X_te_t)
            except Exception:
                explainer = shap.LinearExplainer(core, X_tr_t)
                sv = explainer.shap_values(X_te_t)
            return sv, X_te_t

        if "shap_b" in d:
            sv, X_te_t = d["shap_b"]
        else:
            sv, X_te_t = compute_shap_b(
                d["trained"][d["best_name"]], d["X_te"], d["X_tr"], FEAT_COLS, d["best_name"])

        cls_colors = ["steelblue", "goldenrod", "tomato"]

        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            mean_abs = np.abs(sv).mean(axis=(0, 2))
            class_means = [np.abs(sv[:, :, i]).mean(axis=0) for i in range(sv.shape[2])]
        elif isinstance(sv, list):
            mean_abs = np.mean([np.abs(s) for s in sv], axis=0).mean(axis=0)
            class_means = [np.abs(s).mean(axis=0) for s in sv]
        else:
            mean_abs = np.abs(sv).mean(axis=0)
            class_means = None

        shap_df = pd.DataFrame({"피처": FEAT_COLS, "Mean |SHAP|": mean_abs}).sort_values("Mean |SHAP|")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].barh(shap_df["피처"], shap_df["Mean |SHAP|"],
                     color="skyblue", edgecolor="white", alpha=0.85)
        axes[0].set_title("SHAP 전역 중요도 (클래스 평균)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Mean |SHAP value|")

        if class_means is not None:
            x_pos = np.arange(len(FEAT_COLS))
            w = 0.25
            for i, (cls, color) in enumerate(zip(CLASS_ORDER, cls_colors)):
                axes[1].barh(x_pos + i * w, class_means[i], w,
                             label=cls, color=color, alpha=0.8, edgecolor="white")
            axes[1].set_yticks(x_pos + w)
            axes[1].set_yticklabels(FEAT_COLS, fontsize=9)
            axes[1].set_title("클래스별 SHAP 중요도", fontsize=12, fontweight="bold")
            axes[1].set_xlabel("Mean |SHAP|")
            axes[1].legend()
        plt.suptitle(f"SHAP 피처 중요도 ({d['best_name']})", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
