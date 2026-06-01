import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json, os
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
import folium
from streamlit_folium import st_folium
from .utils import setup_font, load_main, FEAT_COLS, DATA_DIR

RANDOM_STATE = 42
K_OPTIMAL = 5
CLUSTER_NAMES = {
    0: "초고령·인프라포화",
    1: "고령화가속·인프라갭",
    2: "고령화안정·인프라충족",
    3: "저고령·수요태동",
    4: "고령독거·수요폭증",
}
PALETTE = ["blue", "green", "yellow", "orange", "red"]


@st.cache_resource
def prepare_track_c():
    df = load_main()
    X_raw = df[FEAT_COLS].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    km = KMeans(n_clusters=K_OPTIMAL, n_init=50, random_state=RANDOM_STATE)
    df["cluster_km"] = km.fit_predict(X_scaled)
    df["cluster_label"] = df["cluster_km"].map(CLUSTER_NAMES)

    acc_means = df.groupby("cluster_km")["요양수요가속도"].mean()
    sorted_clusters = acc_means.sort_values().index.tolist()

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    # K selection metrics
    k_range = range(2, 11)
    wcss_list, sil_list, db_list = [], [], []
    for k in k_range:
        km_k = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = km_k.fit_predict(X_scaled)
        wcss_list.append(km_k.inertia_)
        sil_list.append(silhouette_score(X_scaled, labels))
        db_list.append(davies_bouldin_score(X_scaled, labels))

    return {
        "df": df, "X_scaled": X_scaled, "X_raw": X_raw,
        "km": km, "pca": pca, "X_pca": X_pca,
        "sorted_clusters": sorted_clusters,
        "k_range": list(k_range),
        "wcss": wcss_list, "sil": sil_list, "db": db_list,
    }


def build_kmeans_map(df, sorted_clusters):
    df["행정동코드_str"] = df["행정동코드"].astype(str)
    geojson_path = os.path.join(DATA_DIR, "HangJeongDong_ver20260201.geojson")
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson_all = json.load(f)
    geojson_seoul = {
        "type": "FeatureCollection",
        "features": [
            feat for feat in geojson_all["features"]
            if str(feat["properties"].get("adm_cd2", ""))[:2] == "11"
        ],
    }
    map_info = df.set_index("행정동코드_str")[[
        "행정동명", "cluster_km", "cluster_label",
        "고령화율", "독거노인비율", "인프라갭", "요양수요가속도",
    ]].copy()

    for feat in geojson_seoul["features"]:
        code = str(feat["properties"].get("adm_cd2", ""))
        if code in map_info.index:
            row = map_info.loc[code]
            p = feat["properties"]
            p["cluster_km"] = int(row["cluster_km"])
            p["cluster_label"] = str(row["cluster_label"])
            p["dong_name"] = str(row["행정동명"])
            p["고령화율"] = round(float(row["고령화율"]), 2)
            p["독거노인비율"] = round(float(row["독거노인비율"]), 2)
            p["인프라갭"] = round(float(row["인프라갭"]), 1)
            p["요양수요가속도"] = round(float(row["요양수요가속도"]), 2)
        else:
            for k in ["cluster_label", "dong_name"]:
                feat["properties"][k] = "미분류"
            feat["properties"]["cluster_km"] = -1
            for k in ["고령화율", "독거노인비율", "인프라갭", "요양수요가속도"]:
                feat["properties"][k] = 0.0

    m = folium.Map(location=[37.555, 126.988], zoom_start=11, tiles="CartoDB dark_matter")

    def style_fn(feature):
        cid = feature["properties"].get("cluster_km", -1)
        color = PALETTE[sorted_clusters.index(cid) % len(PALETTE)] if cid in sorted_clusters else "#374151"
        return {"fillColor": color, "color": "#0f1117", "weight": 0.7, "fillOpacity": 0.82}

    folium.GeoJson(
        geojson_seoul,
        style_function=style_fn,
        highlight_function=lambda x: {"weight": 2.5, "color": "white", "fillOpacity": 0.95},
        tooltip=folium.GeoJsonTooltip(
            fields=["dong_name", "cluster_label", "고령화율", "독거노인비율", "인프라갭", "요양수요가속도"],
            aliases=["행정동:", "클러스터:", "고령화율(%):", "독거노인비율(%):", "인프라갭:", "요양수요가속도:"],
            localize=True, sticky=True,
            style="background-color:#1e2230;color:#e2e8f0;font-size:12px;padding:8px;border-radius:4px;",
        ),
    ).add_to(m)

    cluster_sizes = df["cluster_km"].value_counts()
    legend_items = "".join([
        f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0;'>"
        f"<div style='width:14px;height:14px;background:{PALETTE[i%len(PALETTE)]};border-radius:2px;'></div>"
        f"<span style='font-size:11px;'>C{cid}: {CLUSTER_NAMES.get(cid,'')} ({cluster_sizes.get(cid,0)}개)</span></div>"
        for i, cid in enumerate(sorted_clusters)
    ])
    legend_html = f"""
    <div style='position:fixed;bottom:30px;left:30px;z-index:9999;background:#1e2230;
         border:1px solid #3a4060;border-radius:8px;padding:12px 16px;
         font-family:sans-serif;color:#e2e8f0;min-width:200px;'>
      <b style='font-size:13px;'>K-Means 클러스터 (K={K_OPTIMAL})</b>
      <div style='margin-top:6px;'>{legend_items}</div>
      <div style='margin-top:6px;font-size:10px;color:#94a3b8;'>마우스오버 시 상세 정보</div>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def build_choropleth_map(df):
    df["행정동코드_str"] = df["행정동코드"].astype(str)
    geojson_path = os.path.join(DATA_DIR, "HangJeongDong_ver20260201.geojson")
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson_all = json.load(f)
    geojson_seoul = {
        "type": "FeatureCollection",
        "features": [
            feat for feat in geojson_all["features"]
            if str(feat["properties"].get("adm_cd2", ""))[:2] == "11"
        ],
    }
    for feat in geojson_seoul["features"]:
        code = str(feat["properties"].get("adm_cd2", ""))
        row = df[df["행정동코드_str"] == code]
        if not row.empty:
            feat["properties"]["dong_name"] = row.iloc[0]["행정동명"]
            feat["properties"]["cluster_label"] = row.iloc[0]["cluster_label"]
            feat["properties"]["요양수요가속도"] = round(float(row.iloc[0]["요양수요가속도"]), 2)
        else:
            feat["properties"]["dong_name"] = "미분류"
            feat["properties"]["cluster_label"] = "미분류"
            feat["properties"]["요양수요가속도"] = 0.0

    m = folium.Map(location=[37.555, 126.988], zoom_start=11, tiles="CartoDB dark_matter")
    df_choro = df[["행정동코드_str", "요양수요가속도"]].copy()
    df_choro.columns = ["code", "y_val"]

    folium.Choropleth(
        geo_data=geojson_seoul,
        data=df_choro,
        columns=["code", "y_val"],
        key_on="feature.properties.adm_cd2",
        fill_color="YlOrRd",
        fill_opacity=0.85,
        line_opacity=0.3,
        nan_fill_color="#374151",
        legend_name="요양수요가속도 [ln(독거노인_2023/2018)×100]",
        bins=7,
    ).add_to(m)

    folium.GeoJson(
        geojson_seoul,
        style_function=lambda x: {"fillColor": "transparent", "color": "transparent", "weight": 0},
        tooltip=folium.GeoJsonTooltip(
            fields=["dong_name", "요양수요가속도", "cluster_label"],
            aliases=["행정동:", "요양수요가속도:", "K-Means 클러스터:"],
            localize=True, sticky=True,
            style="background-color:#1e2230;color:#e2e8f0;font-size:12px;padding:8px;border-radius:4px;",
        ),
    ).add_to(m)
    return m


def show():
    setup_font()
    st.title("🗺️ Track C — 고령 취약계층 지역 군집분석")
    st.markdown("**알고리즘**: K-Means (K=5) | **피처**: 10개 지표 (StandardScaler 전처리)")

    with st.spinner("군집 분석 실행 중... (최초 1회만 실행)"):
        d = prepare_track_c()

    df = d["df"]
    sorted_clusters = d["sorted_clusters"]
    cluster_sizes = df["cluster_km"].value_counts()

    # ── 인터랙티브 지도 (상단 고정) ────────────────────────────────────────
    st.header("🗺️ 인터랙티브 지도 시각화")
    st.markdown("행정동에 마우스를 올리면 상세 정보를 확인할 수 있습니다.")

    map_col1, map_col2 = st.columns(2)

    with map_col1:
        st.subheader("K-Means 클러스터 지도")
        for i, cid in enumerate(sorted_clusters):
            color = PALETTE[i % len(PALETTE)]
            st.markdown(
                f"<span style='background:{color};padding:2px 8px;border-radius:4px;"
                f"color:white;font-size:12px;margin:2px;display:inline-block;'>"
                f"C{cid}: {CLUSTER_NAMES[cid]} ({cluster_sizes.get(cid,0)}개)</span>",
                unsafe_allow_html=True,
            )
        with st.spinner("지도 로딩 중..."):
            m_km = build_kmeans_map(df.copy(), sorted_clusters)
            st_folium(m_km, width=680, height=480, returned_objects=[])

    with map_col2:
        st.subheader("요양수요가속도 Choropleth")
        st.markdown(
            "노란색(낮음) → 주황색 → 빨간색(높음) 순으로 "
            "`ln(독거노인_2023/2018)×100` 값을 표현합니다."
        )
        with st.spinner("지도 로딩 중..."):
            m_choro = build_choropleth_map(df.copy())
            st_folium(m_choro, width=680, height=480, returned_objects=[])

    st.divider()

    # ── 세부 분석 탭 ───────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📉 피처 탐색", "🔢 최적 K 선정", "📦 클러스터 분포",
        "🎯 PCA 분석", "📊 클러스터 프로파일",
    ])

    # ── Tab 1: 피처 탐색 ──────────────────────────────────────────────────
    with tab1:
        st.subheader("군집화 피처 분포 (10개 지표)")
        feat_colors = [
            "#3b82f6", "#22c55e", "#f59e0b", "#ec4899", "#8b5cf6",
            "#06b6d4", "#f97316", "#a3e635", "#fb7185", "#c084fc",
        ]
        df_src = load_main()
        fig, axes = plt.subplots(2, 5, figsize=(22, 9))
        axes = axes.flatten()
        for i, (feat, color) in enumerate(zip(FEAT_COLS, feat_colors)):
            ax = axes[i]
            data = df_src[feat].dropna()
            ax.hist(data, bins=28, color=color, edgecolor="white", alpha=0.85, density=True)
            kde = gaussian_kde(data)
            xr = np.linspace(data.min(), data.max(), 200)
            ax.plot(xr, kde(xr), color="white", lw=2)
            ax.axvline(data.mean(), color="#fbbf24", lw=1.8, ls="--", label=f"평균={data.mean():.2f}")
            ax.axvline(data.median(), color="#a78bfa", lw=1.8, ls=":", label=f"중앙={data.median():.2f}")
            ax.set_title(feat, fontsize=10, fontweight="bold")
            ax.legend(fontsize=7)
        plt.suptitle("군집화 피처 분포 (420개 행정동)", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.subheader("피처 간 상관계수 히트맵")
        corr = df_src[FEAT_COLS].corr()
        fig, ax = plt.subplots(figsize=(10, 9))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={"size": 8})
        ax.set_title("피처 간 상관계수 히트맵", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 2: 최적 K ─────────────────────────────────────────────────────
    with tab2:
        st.subheader("최적 K 선정 — Elbow / Silhouette / Davies-Bouldin")
        k_vals = d["k_range"]
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        for ax, (vals, title, color, hint) in zip(axes, [
            (d["wcss"], "WCSS (Inertia) — 엘보우", "steelblue", "↓ 낮을수록 좋음"),
            (d["sil"], "평균 실루엣 점수", "seagreen", "↑ 높을수록 좋음"),
            (d["db"], "Davies-Bouldin 지수", "goldenrod", "↓ 낮을수록 좋음"),
        ]):
            ax.plot(k_vals, vals, "o-", color=color, lw=2.5, markersize=8, zorder=3)
            ax.axvline(K_OPTIMAL, color="tomato", ls="--", lw=2, label=f"선택 K={K_OPTIMAL}")
            opt_val = vals[k_vals.index(K_OPTIMAL)]
            ax.scatter([K_OPTIMAL], [opt_val], s=180, color="tomato", zorder=5)
            ax.text(K_OPTIMAL + 0.1, opt_val, f"  {opt_val:.3f}", fontsize=9, color="tomato")
            ax.set_xticks(k_vals)
            ax.set_xlabel("K (클러스터 수)")
            ax.set_title(f"{title}\n({hint})", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.25)
        plt.suptitle("K-Means 최적 K 선정 (K=5 채택)", fontsize=13, fontweight="bold", y=1.03)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.subheader("실루엣 세부 분석 — K 후보 3개 비교")
        k_cands = [max(2, K_OPTIMAL - 1), K_OPTIMAL, min(10, K_OPTIMAL + 1)]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        tab_colors = plt.cm.tab10.colors
        for ax, K_c in zip(axes, k_cands):
            km_c = KMeans(n_clusters=K_c, n_init=20, random_state=RANDOM_STATE)
            labels_c = km_c.fit_predict(d["X_scaled"])
            sil_v = silhouette_samples(d["X_scaled"], labels_c)
            avg_sil = silhouette_score(d["X_scaled"], labels_c)
            y_lower = 10
            for k in range(K_c):
                cluster_sil = np.sort(sil_v[labels_c == k])
                y_upper = y_lower + cluster_sil.shape[0]
                color = tab_colors[k % 10]
                ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                                  facecolor=color, edgecolor=color, alpha=0.75)
                ax.text(-0.07, y_lower + cluster_sil.shape[0] / 2,
                         f"C{k}\n({cluster_sil.shape[0]})", ha="right", va="center",
                         fontsize=8, color=color)
                y_lower = y_upper + 10
            ax.axvline(avg_sil, color="white", ls="--", lw=2,
                        label=f"평균={avg_sil:.3f}")
            ax.set_xlim([-0.25, 1.0])
            ax.set_ylim([0, y_lower])
            ax.set_yticks([])
            ax.set_xlabel("실루엣 계수")
            is_opt = K_c == K_OPTIMAL
            ax.set_title(f"K = {K_c}" + (" ★선택" if is_opt else ""),
                          fontsize=12, fontweight="bold",
                          color="tomato" if is_opt else "black")
            ax.legend(fontsize=9)
        plt.suptitle("실루엣 분석 — K 후보별 품질 비교", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 3: 클러스터 분포 ─────────────────────────────────────────────
    with tab3:
        st.subheader("피처별 클러스터 간 분포 (Box Plot, 10개 지표)")
        palette_dict = {cid: PALETTE[sorted_clusters.index(cid) % len(PALETTE)]
                        for cid in range(K_OPTIMAL)}
        fig, axes = plt.subplots(2, 5, figsize=(22, 8))
        for ax, feat in zip(axes.flatten(), FEAT_COLS):
            data_by = [df[df["cluster_km"] == cid][feat].values for cid in range(K_OPTIMAL)]
            bp = ax.boxplot(data_by, patch_artist=True,
                            medianprops=dict(color="white", linewidth=2.5),
                            whiskerprops=dict(color="#94a3b8"), capprops=dict(color="#94a3b8"),
                            flierprops=dict(marker="o", markersize=3, alpha=0.5))
            for patch, cid in zip(bp["boxes"], range(K_OPTIMAL)):
                patch.set_facecolor(palette_dict[cid])
                patch.set_alpha(0.8)
            ax.set_xticklabels([f"C{c}" for c in range(K_OPTIMAL)], fontsize=9)
            ax.set_title(feat, fontsize=10, fontweight="bold")
            ax.grid(axis="y", alpha=0.2)
        plt.suptitle("피처별 클러스터 간 분포 비교", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 4: PCA 분석 ───────────────────────────────────────────────────
    with tab4:
        st.subheader("PCA 2D 군집 시각화 + Biplot")
        pca = d["pca"]
        X_pca = d["X_pca"]
        X_scaled = d["X_scaled"]
        var_ratio = pca.explained_variance_ratio_

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # PCA scatter
        ax = axes[0]
        for i, cid in enumerate(sorted_clusters):
            mask = df["cluster_km"] == cid
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=PALETTE[i % len(PALETTE)], s=35, alpha=0.75, edgecolors="none",
                        label=f"C{cid}: {CLUSTER_NAMES[cid]} (n={mask.sum()})")
        centers_pca = pca.transform(d["km"].cluster_centers_)
        for i, cid in enumerate(range(K_OPTIMAL)):
            rank_i = sorted_clusters.index(cid)
            ax.scatter(centers_pca[cid, 0], centers_pca[cid, 1],
                        c=PALETTE[rank_i % len(PALETTE)], s=250, marker="*",
                        edgecolors="white", linewidths=1.5, zorder=5)
            ax.annotate(f"C{cid}", (centers_pca[cid, 0], centers_pca[cid, 1]),
                         fontsize=10, fontweight="bold", xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=11)
        ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=11)
        ax.set_title("PCA 2D — K-Means 클러스터 (★=센트로이드)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

        # Biplot
        ax2 = axes[1]
        for i, cid in enumerate(sorted_clusters):
            mask = df["cluster_km"] == cid
            ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                         c=PALETTE[i % len(PALETTE)], s=25, alpha=0.5, edgecolors="none")
        loadings = pca.components_.T
        scale = 3.0
        for j, feat in enumerate(FEAT_COLS):
            ax2.annotate("", xy=(loadings[j, 0]*scale, loadings[j, 1]*scale), xytext=(0, 0),
                          arrowprops=dict(arrowstyle="->", color="#fbbf24", lw=2))
            ax2.text(loadings[j, 0]*scale*1.15, loadings[j, 1]*scale*1.15,
                      feat, fontsize=9, color="#fbbf24", fontweight="bold", ha="center")
        ax2.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=11)
        ax2.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=11)
        ax2.set_title("PCA Biplot — 피처 로딩 방향", fontsize=12, fontweight="bold")
        ax2.axhline(0, color="#4b5563", lw=0.8, ls="--")
        ax2.axvline(0, color="#4b5563", lw=0.8, ls="--")
        ax2.grid(alpha=0.2)

        plt.suptitle(f"PCA 2D (PC1+PC2 = {sum(var_ratio)*100:.1f}% 분산 설명)",
                      fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 5: 클러스터 프로파일 ─────────────────────────────────────────
    with tab5:
        st.subheader("클러스터별 피처 프로파일 히트맵 (Z-score)")
        cluster_means = df.groupby("cluster_km")[FEAT_COLS].mean()
        cluster_means_z = cluster_means.apply(lambda col: (col - col.mean()) / (col.std() + 1e-9))
        cluster_means_z.index = [f"C{cid}\n{CLUSTER_NAMES[cid]}" for cid in cluster_means_z.index]

        fig, ax = plt.subplots(figsize=(11, 6))
        sns.heatmap(cluster_means_z, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                     linewidths=1.5, ax=ax, annot_kws={"size": 11, "weight": "bold"},
                     cbar_kws={"label": "Z-score"}, vmin=-2, vmax=2)
        ax.set_title("클러스터별 피처 Z-score 프로파일\n빨간색=상대적 높음, 파란색=상대적 낮음",
                      fontsize=12, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.subheader("클러스터별 요양수요가속도 통계")
        y_stats = df.groupby("cluster_km")["요양수요가속도"].agg(
            ["mean", "std", "min", "max", "count"]).round(2)
        y_stats.index = [f"C{i}: {CLUSTER_NAMES[i]}" for i in y_stats.index]
        st.dataframe(y_stats, use_container_width=True)
