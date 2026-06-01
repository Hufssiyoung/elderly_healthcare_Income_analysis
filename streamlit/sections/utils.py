import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

FEAT_COLS = [
    "고령화율", "고령화율_변화", "독거노인비율",
    "비율_75-79세", "비율_80-84세", "비율_85세이상",
    "인프라갭", "포화도", "시설수", "수요공급갭지수",
]
TARGET = "요양수요가속도"
CLASS_ORDER = ["저위험", "중위험", "고위험"]


def setup_font():
    fonts = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fp in fonts:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)
            plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110


@st.cache_data
def load_main():
    return pd.read_csv(os.path.join(DATA_DIR, "df_analysis.csv"), encoding="utf-8-sig")


@st.cache_data
def load_criteria():
    return pd.read_csv(os.path.join(DATA_DIR, "df_criteria.csv"), encoding="utf-8-sig")


@st.cache_data
def load_merged():
    df = load_main()
    dc = load_criteria()
    merged = df.merge(
        dc[["행정동코드", "위험등급", "IMD_score"]],
        on="행정동코드", how="inner"
    )
    merged["위험등급"] = pd.Categorical(
        merged["위험등급"], categories=CLASS_ORDER, ordered=True
    )
    return merged
