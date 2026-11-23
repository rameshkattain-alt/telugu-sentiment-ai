from google import genai
import pandas as pd
import streamlit as st

API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

st.set_page_config(page_title="Telugu & Tenglish Sentiment AI", page_icon="🇮🇳")
st.title("🇮🇳 తెలుగు & Tenglish Sentiment AI")
st.write("తెలుగు & తంగ్లీష్ కామెంట్లలోని Positive, Negative, మరియు Sarcasm గుర్తించే AI.")


def get_sentiment(text):
  prompt = f"""
    Analyze this Telugu/Tenglish comment: "{text}"
    Classify strictly into one word: Positive, Negative, or Sarcasm.
    Then on a new line, give a short reason in English and Telugu.
    """
  available_models = [
      m.name.replace("models/", "")
      for m in client.models.list()
      if "generateContent" in m.supported_actions
      and ("gemini" in m.name or "gemma" in m.name)
  ]
  for model_name in available_models:
    try:
      res = client.models.generate_content(model=model_name, contents=prompt)
      return res.text
    except Exception:
      continue
  return "Error: Server Busy"


tab1, tab2 = st.tabs(["💬 సింగిల్ కామెంట్ చెక్", "📊 బల్క్ CSV & చార్ట్"])

with tab1:
  comment = st.text_input(
      "కామెంట్ ఇక్కడ టైప్ చేయండి:",
      placeholder="ఉదాహరణకు: Super anna, full support",
  )
  if st.button("చెక్ చెయ్ (Check)"):
    if comment:
      with st.spinner("AI ఆలోచిస్తోంది..."):
        result = get_sentiment(comment)
        st.success("ఫలితం సిద్ధం!")
        st.write(result)
    else:
      st.warning("ముందు ఏదైనా కామెంట్ టైప్ చేయండి!")

with tab2:
  st.write("అనేక కామెంట్లు ఉన్న CSV ఫైల్ అప్‌లోడ్ చేయండి ('comment' అనే కాలమ్ ఉండాలి).")
  uploaded_file = st.file_uploader("CSV ఫైల్ ఎంచుకోండి", type=["csv"])
  if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "comment" in df.columns:
      if st.button("అన్ని కామెంట్లను విశ్లేషించు"):
        categories = []
        progress = st.progress(0)
        for i, row in df.iterrows():
          ans = get_sentiment(str(row["comment"]))
          first_line = ans.split("\n").strip()
          if "Sarcasm" in first_line:
            categories.append("Sarcasm")
          elif "Negative" in first_line:
            categories.append("Negative")
          else:
            categories.append("Positive")
          progress.progress((i + 1) / len(df))

        df["Sentiment"] = categories
        st.dataframe(df)
        st.subheader("📈 సెంటిమెంట్ విశ్లేషణ గ్రాఫ్")
        st.bar_chart(df["Sentiment"].value_counts())
    else:
      st.error("CSV ఫైల్‌లో 'comment' అనే కాలమ్ ఉండాలి!")
