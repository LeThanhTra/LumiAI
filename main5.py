# -----------------------------
# 1. Kết nối API
# -----------------------------
import pandas as pd
import streamlit as st
import faiss
import numpy as np
import pickle
import json
import os
import unicodedata
import re

from openai import OpenAI
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_API_KEY"] = "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"

client = OpenAI()
#client = OpenAI(
   # base_url="https://openrouter.ai/api/v1",
   # api_key="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"
#)


# -----------------------------
# CACHE CSV
# -----------------------------
@st.cache_data
def load_csv():
    df = pd.read_csv("data.csv")
    df["Loại hình list"] = df["Loại hình"].apply(
        lambda x: [i.strip() for i in x.split("/")]
    )
    df["text"] = df.apply(
        lambda x: f"{x['Tên địa điểm']} | {', '.join(x['Loại hình list'])}",
        axis=1
    )
    return df

df = load_csv()


# -----------------------------
# CACHE FAISS & PICKLE
# -----------------------------
@st.cache_resource
def load_faiss_and_chunks():
    fact_index = faiss.read_index("vector_index.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return fact_index, chunks, metadata

fact_index, chunks, metadata = load_faiss_and_chunks()


# -----------------------------
# 3. Preference functions
# -----------------------------
PREF_FILE = "user_preferences.json"

def load_prefs():
    if not os.path.exists(PREF_FILE):
        return {}
    return json.load(open(PREF_FILE, "r", encoding="utf-8"))

def save_prefs(p):
    json.dump(p, open(PREF_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# -----------------------------
# UI – chọn sở thích
# -----------------------------
st.title("LumiAI – Thiết lập sở thích du lịch")

travel_types = ["Tôn giáo","Sinh thái","Dịch vụ","Miệt vườn","Kiến trúc","Văn hoá","Tâm linh","Di tích"]

interests = st.multiselect("Chọn loại hình du lịch:", travel_types)

# -----------------------------
# Lưu sở thích + Gợi ý ngay
# -----------------------------

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # chuẩn hóa unicode, tách dấu, bỏ dấu (để so sánh không phân biệt dấu)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")  # remove accents
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)   # thay ký tự đặc biệt bằng space
    s = re.sub(r"\s+", " ", s).strip()
    return s


@st.cache_data
def normalize_categories(df):
    def split_and_normalize(cat):
        parts = [p.strip() for p in str(cat).split("/")]
        return [normalize_text(p) for p in parts]
    df["Loại hình list norm"] = df["Loại hình"].apply(split_and_normalize)
    return df

df = normalize_categories(df)


prefs = load_prefs()
pref_norms = [i.lower() for i in prefs["interests"]]

def is_match(row_cats, pref_norms):
    return len(set(row_cats) & set(pref_norms)) > 0

df["match"] = df["Loại hình list norm"].apply(lambda cats: is_match(cats, pref_norms))

df_matched = df[df["match"] == True]
df_sorted = df_matched  # nếu không muốn xếp hạng thì bỏ Sorting


# --- UI handler ---
if st.button("Lưu sở thích"):
    user_pref = {
        "interests": interests  # lưu nguyên gốc để hiển thị
    }
    # Lưu cả dạng chuẩn hóa để so sánh nhanh
    user_pref["_interests_norm"] = [normalize_text(i) for i in interests]
    save_prefs(user_pref)
    st.success("Đã lưu thông tin sở thích!")

    # Chuẩn hóa cột Loại hình list 1 lần (an toàn nếu chạy nhiều lần)
    def split_and_normalize(cat):
        parts = [p.strip() for p in str(cat).split("/")]
        return [normalize_text(p) for p in parts if p]

    df["Loại hình list norm"] = df["Loại hình"].apply(split_and_normalize)

    prefs = load_prefs()
    pref_norms = prefs.get("_interests_norm", [])

    def match_score(row_norms, pref_norms):
        # dùng giao (intersection) trên tập để cho điểm công bằng
        set_row = set(row_norms)
        set_pref = set(pref_norms)
        return len(set_row & set_pref)

    # tính score dựa trên cột chuẩn hóa
    df["score"] = df["Loại hình list norm"].apply(lambda r: match_score(r, pref_norms))
    df_sorted = df[df["score"] > 0].sort_values(by="score", ascending=False)

    # nếu muốn show top 10, nhưng đảm bảo chỉ lấy những hàng có score>0
    top_df = df_sorted.head(10)
    if top_df.empty:
        st.info("Không tìm thấy địa điểm phù hợp với sở thích của bạn trong dataset.")
    else:
        location_summaries = "\n\n".join(top_df["text"].tolist())

        # gọi model (giữ nguyên phần gọi API của anh nếu muốn)
        # CHÚ Ý BẢO MẬT: KHÔNG đặt API KEY cứng trong mã nguồn; dùng biến môi trường hoặc secret manager.
        system_prompt = f"""
You are LumiAI — an AI for tourism, culture, and fact verification.
Your tasks:
User travel preferences:
    {user_pref}
-Using the CSV dataset:
    +Identify related locations
    +Provide travel recommendations
    +Describe features, activities, and experiences
"""
        user_message = (
            "Dữ liệu (chỉ sử dụng những dòng bên dưới để trả lời):\n\n"
            f"{location_summaries}\n\n"
            "Yêu cầu: Dựa trên dữ liệu ở trên, hãy liệt kê gợi ý du lịch sát với sở thích người dùng (không thêm thừa), mô tả ngắn mỗi nơi (2-3 câu)."
        )

        completion = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        suggestion = completion.choices[0].message.content
        st.session_state.suggestion = suggestion
        # Lưu lịch sử gợi ý
        if "suggestions_history" not in st.session_state:
            st.session_state.suggestions_history = []

        st.session_state.suggestions_history.append(suggestion)



if "suggestion" in st.session_state:
    st.markdown("### ✨ Gợi ý du lịch từ LumiAI")
    st.write(st.session_state.suggestion)

suggestion = st.session_state.get("suggestion", "")
# -----------------------------
# Chat liên tục
# -----------------------------
st.write("---")
st.subheader("💬 Hỏi LumiAI thêm:")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Nhập yêu cầu của bạn...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    prefs = load_prefs()

    # Lấy embedding
    query_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=user_input
    ).data[0].embedding

    q = np.array([query_emb]).astype("float32")


    #D1, I1 = suggestion.search(q, k=10)
    #valid_indices = [i for i in I1[0] if 0 <= i < len(df)]
    #retrieved_csv = "\n".join(df.iloc[i]["text"] for i in valid_indices)

    # Tìm trong CHUNKS
    D2, I2 = fact_index.search(q, k=10)
    retrieved_chunks = ""
    for i in I2[0]:
        if 0 <= i < len(chunks):
            m = metadata[i]
            retrieved_chunks += f"\n\n[Chunk {i} — Lines {m['start_line']}–{m['end_line']}]:\n{chunks[i]}"

    # System prompt chat
    system_prompt = f"""
You are LumiAI — an AI for tourism, culture, and fact verification.
Your tasks:
-Using the CSV dataset:
    +Identify related locations
    +Provide travel recommendations
    +Describe features, activities, and experiences
-Using the CHUNKS dataset:
    + Deliver cultural, historical, and mythological explanations
    + Include chunk, lines, page (the number which stands alone after the information) after the response
    + Always include the exact source for every statement.
    + If multiple statements use multiple chunks,pages, list all sources
    + Never invent sources
    + Verify correctness if the question requires fact-checking
    + Absolutely no fabrication of facts.
    + If the data is missing → respond with “Không có dữ liệu”.
    + Add link, source for the tour

You operate in CLOSED-BOOK MODE.
You are ONLY allowed to speak using information that appears in the datasets below.
You are forbidden from using world knowledge, memory, training data, or assumptions.
If the answer requires any information that is not explicitly present in the dataset,
you must respond with EXACTLY: “Không có dữ liệu”.
You must quote the exact chunk + line numbers for every fact used.

Suggestion message, you need to use if user ask according to your previous suggestion:
{suggestion}

Cultural Data:
{retrieved_chunks}
"""

    completion = client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=[
            {"role": "system", "content": system_prompt},
            *st.session_state.messages[-5:],  # giữ lịch sử ngắn
            {"role": "user", "content": user_input}
        ]
    )

    ai_msg = completion.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": ai_msg})

    with st.chat_message("assistant"):
        st.markdown(ai_msg)