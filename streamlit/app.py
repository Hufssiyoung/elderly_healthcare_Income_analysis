import streamlit as st
import streamlit.components.v1 as components
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_HTML = os.path.join(BASE_DIR, "프로젝트_계획_수정본.html")

st.set_page_config(
    page_title="고령 취약계층 헬스케어 분석",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

from sections import eda, track_a, track_b, track_c

with st.sidebar:
    st.title("🏥 헬스케어 분석")
    st.markdown("**고령 취약계층 수요 예측 플랫폼**")
    st.markdown("서울시 420개 행정동 | 2023년 기준")
    st.divider()

    page = st.radio(
        "분석 섹션",
        options=[
            "🏠 프로젝트 개요",
            "📊 EDA",
            "📈 Track A — 회귀분석",
            "🎯 Track B — 분류분석",
            "🗺️ Track C — 군집분석",
        ],
    )

    st.divider()
    st.caption("X 변수 10개 | Y: 요양수요가속도\n데이터: df_analysis.csv")

if page == "🏠 프로젝트 개요":
    with open(PROJECT_HTML, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=7000, scrolling=True)
elif page == "📊 EDA":
    eda.show()
elif page == "📈 Track A — 회귀분석":
    track_a.show()
elif page == "🎯 Track B — 분류분석":
    track_b.show()
elif page == "🗺️ Track C — 군집분석":
    track_c.show()
