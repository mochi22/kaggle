import streamlit as st
from kaggle_utils import get_active_competitions


st.set_page_config(page_title="Kaggle AI Helper", layout="centered")
st.title("🧠 Kaggleコンペ選択")

# コンペ一覧取得
st.write("Kaggleから現在アクティブなコンペを取得中...")
try:
    competitions = get_active_competitions()
except Exception as e:
    st.error(f"取得失敗: {e}")
    st.stop()

# 選択UI
options = [f"{comp['title']} ({comp['ref']}) - 締切: {comp['deadline']}" for comp in competitions]
selected = st.selectbox("👇 参加するコンペを選んでください", options)

# 選ばれたコンペのrefを取得
selected_ref = competitions[options.index(selected)]['ref']
st.success(f"✅ 選択されたコンペ: `{selected_ref}`")

from kaggle_utils import get_and_save_competition_html, get_and_save_competition_discussion

# get overview html
# over_view_html = get_competition_description(selected_ref)

# saved_file_path=get_and_save_competition_html(selected_ref, "overview", save_path='datas/competitions')
# st.subheader("📄 コンペの概要")
# st.write(f"コンペの概要はHTMLファイルとして保存しました。ファイルパス: {saved_file_path}")


# save sqlite
# save_competition_to_db(selected_ref, over_view_html)
 
saved_file_path=get_and_save_competition_html(selected_ref, "discussion", save_path='datas/discussions')
st.subheader("📄 コンペの概要")
st.write(f"discussionはHTMLファイルとして保存しました。ファイルパス: {saved_file_path}")

