import json
from google import genai
import pandas as pd
import streamlit as st

API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

st.set_page_config(page_title="Telugu & Tenglish Sentiment AI", page_icon="🇮🇳")
st.title("🇮🇳 తెలుగు & Tenglish Sentiment AI")
st.write("తెలుగు & తంగ్లీష్ కామెంట్లలోని Positive, Negative, మరియు Sarcasm గుర్తించే AI.")


def call_gemini(prompt):
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
  return None


tab1, tab2 = st.tabs(["💬 సింగిల్ కామెంట్ చెక్", "📊 బల్క్ CSV & చార్ట్"])

with tab1:
  comment = st.text_input(
      "కామెంట్ ఇక్కడ టైప్ చేయండి:",
      placeholder="ఉదాహరణకు: Super anna, full support",
  )
  if st.button("చెక్ చెయ్ (Check)"):
    if comment:
      with st.spinner("AI ఆలోచిస్తోంది..."):
        prompt = f"""
                Analyze this Telugu/Tenglish comment: "{comment}"
                Classify strictly into: Positive, Negative, or Sarcasm.
                Explain the reason clearly in English and Telugu.
                """
        result = call_gemini(prompt)
        if result:
          st.success("ఫలితం సిద్ధం!")
          st.write(result)
        else:
          st.error("ప్రస్తుతం సర్వర్ బిజీగా ఉంది, కాసేపటి తర్వాత ప్రయత్నించండి.")
    else:
      st.warning("ముందు ఏదైనా కామెంట్ టైప్ చేయండి!")

with tab2:
  st.write("అనేక కామెంట్లు ఉన్న CSV ఫైల్ అప్‌లోడ్ చేయండి ('comment' అనే కాలమ్ ఉండాలి).")
  uploaded_file = st.file_uploader("CSV ఫైల్ ఎంచుకోండి", type=["csv"])
  if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "comment" in df.columns:
      if st.button("అన్ని కామెంట్లను విశ్లేషించు"):
        with st.spinner("అన్ని కామెంట్లను ఒకేసారి విశ్లేషిస్తోంది... 5 సెకన్లు వేచి ఉండండి"):
          comments_list = df["comment"].astype(str).tolist()
          batch_prompt = f"""
                    Classify each of the following {len(comments_list)} Telugu/Tenglish comments strictly into one of these three labels: "Positive", "Negative", or "Sarcasm".
                    Return ONLY a valid JSON list of strings in the exact same order, with no markdown formatting or extra text.
                    Comments: {json.dumps(comments_list, ensure_ascii=False)}
                    """
          raw_res = call_gemini(batch_prompt)
          if raw_res:
            try:
              clean_json = (
                  raw_res.replace("```json", "").replace("```", "").strip()
              )
              categories = json.loads(clean_json)
              if len(categories) == len(df):
                df["Sentiment"] = categories
                st.success("విశ్లేషణ పూర్తయింది!")
                st.dataframe(df)
                st.subheader("📈 సెంటిమెంట్ విశ్లేషణ గ్రాఫ్")
                st.bar_chart(df["Sentiment"].value_counts())
              else:
                st.error("దయచేసి మళ్ళీ ఒకసారి బటన్ నొక్కండి.")
            except Exception:
              st.error("ఫలితాన్ని ప్రాసెస్ చేయడంలో చిన్న సమస్య వచ్చింది, మళ్ళీ నొక్కండి.")
          else:
            st.error("సర్వర్ బిజీగా ఉంది, 10 సెకన్ల తర్వాత మళ్ళీ ప్రయత్నించండి.")
    else:
      st.error("CSV ఫైల్‌లో 'comment' అనే కాలమ్ ఉండాలి!")
