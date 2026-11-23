from google import genai
import streamlit as st

API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

st.title("🇮🇳 తెలుగు & Tenglish Sentiment AI")
st.write("కింద కామెంట్ రాయండి. అది Positive, Negative, లేదా వ్యంగ్యమో (Sarcasm) AI చెబుతుంది.")

comment = st.text_input(
    "కామెంట్ ఇక్కడ టైప్ చేయండి:",
    placeholder="ఉదాహరణకు: Super anna, full support",
)

if st.button("చెక్ చెయ్ (Check)"):
  if comment:
    with st.spinner("AI ఆలోచిస్తోంది..."):
      try:
        prompt = f"""
                Analyze this Telugu or Tenglish comment: "{comment}"
                Tell me if it is Positive, Negative, or Sarcasm. 
                Explain the reason in simple Telugu and English.
                """
        # మీ కీకి పనిచేసే అన్ని మోడల్స్‌ను ఆటోమేటిక్‌గా వెతికి మొదటి పనిచేసే మోడల్ వాడుతుంది
        available_models = [
            m.name.replace("models/", "")
            for m in client.models.list()
            if "generateContent" in m.supported_actions
            and ("gemini" in m.name or "gemma" in m.name)
        ]

        success = False
        last_err = ""
        for model_name in available_models:
          try:
            response = client.models.generate_content(
                model=model_name, contents=prompt
            )
            st.success(f"ఫలితం సిద్ధం! (Model: {model_name})")
            st.write(response.text)
            success = True
            break
          except Exception as e:
            last_err = str(e)

        if not success:
          st.error(
              f"అందుబాటులో ఉన్న మోడల్స్: {available_models} | ఎర్రర్: {last_err}"
          )
      except Exception as e:
        st.error(f"ఎర్రర్: {e}")
  else:
    st.warning("ముందు ఏదైనా కామెంట్ టైప్ చేయండి!")
