import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, linregress
from sklearn.preprocessing import StandardScaler, PowerTransformer
from .utils import setup_font, load_main, load_merged, FEAT_COLS, TARGET, CLASS_ORDER


def show():
    setup_font()
    st.title("📊 탐색적 데이터 분석 (EDA)")
    st.markdown("서울시 420개 행정동의 X 변수(10개) 및 Y 변수(`요양수요가속도`) 세부 탐색 자료입니다.")

    df = load_main()
    df_b = load_merged()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📉 변수 분포", "🔍 이상치 분석", "🔗 X↔Y 산점도", "🔄 Y 변환 비교"]
    )

    # ── Tab 1: X 변수 분포 ──────────────────────────────────────────────────
    with tab1:
        st.subheader("X 변수 분포 (10개 지표)")
        feat_units = [
            "(%)", "(%p/년)", "(%)", "(%)", "(%)", "(%)",
            "(갭)", "(0~1)", "(개)", "(지수)",
        ]
        feat_colors = [
            "#3b82f6", "#22c55e", "#f59e0b", "#ec4899", "#8b5cf6",
            "#06b6d4", "#f97316", "#a3e635", "#fb7185", "#c084fc",
        ]
        fig, axes = plt.subplots(2, 5, figsize=(22, 8))
        axes = axes.flatten()
        for i, (feat, color, unit) in enumerate(zip(FEAT_COLS, feat_colors, feat_units)):
            ax = axes[i]
            data = df[feat].dropna()
            ax.hist(data, bins=28, color=color, edgecolor="white", alpha=0.85, density=True)
            kde = gaussian_kde(data)
            xr = np.linspace(data.min(), data.max(), 200)
            ax.plot(xr, kde(xr), color="white", lw=2)
            ax.axvline(data.mean(), color="#fbbf24", lw=1.8, ls="--",
                       label=f"평균={data.mean():.2f}")
            ax.axvline(data.median(), color="#a78bfa", lw=1.8, ls=":",
                       label=f"중앙값={data.median():.2f}")
            ax.set_title(f"{feat} {unit}\n왜도={data.skew():.2f}", fontsize=10, fontweight="bold")
            ax.legend(fontsize=7)
        plt.suptitle("X 변수 분포 (히스토그램 + KDE)", fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.subheader("기술 통계량")
        st.dataframe(df[FEAT_COLS + [TARGET]].describe().round(3), width='stretch')

    # ── Tab 2: 이상치 분석 ─────────────────────────────────────────────────
    with tab2:
        st.subheader("X 변수 이상치 비교 (표준화 후 박스플롯)")
        df_scaled = pd.DataFrame(
            StandardScaler().fit_transform(df[FEAT_COLS].dropna()),
            columns=FEAT_COLS,
        )
        fig, ax = plt.subplots(figsize=(14, 5))
        df_scaled.boxplot(
            ax=ax, vert=True,
            boxprops=dict(color="steelblue"),
            medianprops=dict(color="crimson", linewidth=2),
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
        )
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_title("X 변수 이상치 비교 (Z-score 표준화 후)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Z-score")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.subheader("IQR 기반 이상치 현황")
        outlier_rows = []
        for col in FEAT_COLS:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = ((df[col] < lo) | (df[col] > hi)).sum()
            outlier_rows.append({
                "변수": col, "Q1": round(q1, 2), "Q3": round(q3, 2),
                "하한": round(lo, 2), "상한": round(hi, 2),
                "이상치수": n_out, "이상치율(%)": round(n_out / len(df) * 100, 1),
            })
        st.dataframe(pd.DataFrame(outlier_rows).set_index("변수"), width='stretch')

    # ── Tab 3: X↔Y 산점도 ─────────────────────────────────────────────────
    with tab3:
        st.subheader("X 변수 vs Y(요양수요가속도) 산점도")
        corr_y = df[FEAT_COLS].corrwith(df[TARGET]).sort_values(key=abs, ascending=False)

        col_left, col_right = st.columns([1, 3])
        with col_left:
            selected = st.selectbox(
                "X 변수 선택",
                FEAT_COLS,
                index=0,
                help="Y와의 상관계수 절댓값 순으로 정렬됩니다.",
            )
            st.dataframe(
                corr_y.rename("r값").to_frame().style.format("{:.3f}"),
                width='stretch',
            )
        with col_right:
            x_vals = df[selected]
            y_vals = df[TARGET]
            mask = x_vals.notna() & y_vals.notna()
            slope, intercept, r, p, _ = linregress(x_vals[mask], y_vals[mask])
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(x_vals[mask], y_vals[mask], alpha=0.5, s=25, color="steelblue")
            x_line = np.linspace(x_vals[mask].min(), x_vals[mask].max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color="crimson", lw=2)
            ax.set_xlabel(selected)
            ax.set_ylabel(TARGET)
            ax.set_title(f"{selected} vs Y   (r={r:.3f}, p={p:.4f})", fontsize=12, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.divider()
        st.subheader("상관 상위 4개 X 변수 — 동시 비교")
        top4 = corr_y.abs().nlargest(4).index.tolist()
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, col in zip(axes.flatten(), top4):
            x_v, y_v = df[col], df[TARGET]
            m = x_v.notna() & y_v.notna()
            s, ic, r, p, _ = linregress(x_v[m], y_v[m])
            ax.scatter(x_v[m], y_v[m], alpha=0.45, s=20, color="steelblue")
            xl = np.linspace(x_v[m].min(), x_v[m].max(), 100)
            ax.plot(xl, s * xl + ic, color="crimson", lw=2)
            ax.set_xlabel(col)
            ax.set_ylabel(TARGET)
            ax.set_title(f"{col} (r={r:.3f}, p={p:.4f})", fontsize=10, fontweight="bold")
        plt.suptitle("상관 상위 4개 X 변수 vs Y 산점도 (회귀선 포함)", fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 4: Y 변환 비교 ─────────────────────────────────────────────────
    with tab4:
        st.subheader("Y 변수 — 원본 vs Yeo-Johnson 변환 비교")
        y_vals = df[TARGET].dropna()
        pt = PowerTransformer(method="yeo-johnson")
        y_tf = pt.fit_transform(y_vals.values.reshape(-1, 1)).flatten()

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].hist(y_vals, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
        axes[0].set_title(f"Y 원본   (왜도={y_vals.skew():.2f})", fontsize=11)
        axes[0].set_xlabel(TARGET)

        axes[1].hist(y_tf, bins=30, color="tomato", edgecolor="white", alpha=0.8)
        axes[1].set_title(f"Yeo-Johnson 변환 후   (왜도={pd.Series(y_tf).skew():.2f})", fontsize=11)
        axes[1].set_xlabel("Transformed Y")

        from scipy.stats import probplot
        probplot(y_vals, dist="norm", plot=axes[2])
        axes[2].set_title("Q-Q Plot (정규성 확인)", fontsize=11)

        plt.suptitle("Y 변수(요양수요가속도) 원본 vs 변환 비교", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            f"원본 왜도 **{y_vals.skew():.3f}** → 변환 후 왜도 **{pd.Series(y_tf).skew():.3f}**  \n"
            "왜도 감소 효과가 미미하여 Track A에서는 **원본 Y 그대로** 사용."
        )
