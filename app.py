import streamlit as st
import requests
import pandas as pd
import gspread
from openai import OpenAI
from google.oauth2.service_account import Credentials
from urllib.parse import urlparse

# === 🔐 Load from secrets ===
GOOGLE_API_KEY = st.secrets["google"]["api_key"]
GOOGLE_CX = st.secrets["google"]["search_engine_id"]  # Main medical CSE
SOCIAL_GOOGLE_CX = st.secrets["google"]["SOCIAL_GOOGLE_CX"]  # Social media CSE
OPENAI_API_KEY = st.secrets["openai_api_key"]
GOOGLE_SHEET_NAME = st.secrets["google"]["sheet_name"]
GCP_SERVICE_ACCOUNT = st.secrets["gcp_service_account"]

# === 🤖 OpenAI Client ===
client = OpenAI(api_key=OPENAI_API_KEY)

# === Trusted Medical Sources ===
TRUSTED_SITES = [
    "site:nhs.uk", "site:nih.gov", "site:mayoclinic.org", "site:who.int",
    "site:cdc.gov", "site:clevelandclinic.org", "site:health.harvard.edu",
    "site:pubmed.ncbi.nlm.nih.gov", "site:webmd.com", "site:medlineplus.gov"
]

# === Trust Score Function ===
def compute_trust_score(link, snippet):
    domain = urlparse(link).netloc.lower()

    if any(site in domain for site in ["nhs.uk", "cdc.gov", "who.int", "mayoclinic.org", "clevelandclinic.org"]):
        score = 5
    elif any(site in domain for site in ["gov", "edu", "health.harvard.edu"]):
        score = 4.5
    elif "webmd.com" in domain or "medlineplus.gov" in domain:
        score = 4
    elif "pubmed" in domain:
        score = 3.5
    else:
        score = 3

    if any(year in snippet for year in ["2024", "2023", "2022"]):
        score += 0.5

    return min(score, 5.0)

# === Google Search ===
def get_medical_snippets(query, num_results=5):
    domain_query = " OR ".join(TRUSTED_SITES)
    full_query = f"{query} ({domain_query})"
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CX, "q": full_query, "num": num_results}
    try:
        response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        response.raise_for_status()
        items = response.json().get("items", [])
        items.sort(key=lambda x: 0 if "nhs.uk" in x.get("link", "") else 1)

        results = []
        for item in items:
            title = item["title"]
            link = item["link"]
            snippet = item["snippet"]
            score = compute_trust_score(link, snippet)
            results.append((title, link, snippet, score))
        return results
    except Exception:
        return []

# === ChatGPT Answering ===
def answer_medical_question(question):
    snippets = get_medical_snippets(question)
    if not snippets:
        return "Sorry, no reliable sources available now.", []

    context = "\n".join(f"- **{title}**: {snippet}" for title, link, snippet, score in snippets)
    sources = [(title, link, snippet, score) for title, link, snippet, score in snippets]

    prompt = f"""
Answer clearly using snippets below.
Mention both common and serious conditions if symptoms provided.
End with: "Talk to a doctor to be sure."

Snippets:
{context}

Question: {question}

Answer:
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content.strip()
        return answer + "\n\n**Disclaimer:** Always consult your healthcare provider.", sources
    except Exception as e:
        return f"OpenAI API Error: {e}", []

# === Proactive Advisories ===
RISK_SNIPPETS = {
    "antibiotics": "Misuse of antibiotics can lead to antibiotic resistance.",
    "vaccines": "Vaccines do not cause autism; they are safe and thoroughly tested.",
    "ibuprofen": "Long-term use of ibuprofen may cause kidney or stomach problems.",
    "detox": "Your body detoxifies naturally; detox teas or regimens are often unnecessary and risky.",
    "fatigue": "Persistent fatigue might signal anemia, thyroid issues, or depression.",
    "vision loss": "Sudden vision loss is a medical emergency. Seek immediate care.",
    "headache": "Sudden severe headache could mean stroke. Don’t delay medical help.",
    "chest pain": "Chest pain might indicate a heart attack. Go to the ER immediately.",
    "rash": "If rash is accompanied by fever or trouble breathing, see a doctor quickly."
}

def get_risk_snippets(query):
    return [snippet for keyword, snippet in RISK_SNIPPETS.items() if keyword in query.lower()]

# === Severity Categorization ===
SEVERITY_KEYWORDS = {
    "🔴 Immediate": ["chest pain", "vision loss", "stroke", "aneurysm", "severe headache"],
    "🟠 Urgent": ["high fever", "severe pain", "vomiting", "sudden dizziness"],
    "🟢 Routine": []
}

def classify_severity(query):
    q = query.lower()
    for level, words in SEVERITY_KEYWORDS.items():
        if any(w in q for w in words):
            return level
    return "🟢 Routine"

# === Social Media Search ===
SOCIAL_MEDIA_SITES = ["site:reddit.com", "site:healthunlocked.com"]

def get_social_snippets(query, num_results_per_site=5):
    snippets = []
    for site in SOCIAL_MEDIA_SITES:
        full_query = f"{query} ({site})"
        params = {"key": GOOGLE_API_KEY, "cx": SOCIAL_GOOGLE_CX, "q": full_query, "num": num_results_per_site}
        try:
            response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
            response.raise_for_status()
            for item in response.json().get("items", []):
                title = item["title"]
                link = item["link"]
                snippet = item["snippet"]
                score = compute_trust_score(link, snippet) - 1
                snippets.append((title, link, snippet, max(score, 1.0)))
        except Exception:
            continue
    return snippets

# === Streamlit UI ===
st.set_page_config(page_title="AI Medical Assistant", page_icon="🩺", layout="centered")
st.title("🩺 AI-Powered Medical Assistant")

if "history" not in st.session_state:
    st.session_state.history = []
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

user_age = st.sidebar.text_input("Your Age (optional)")
user_gender = st.sidebar.selectbox("Your Gender (optional)", ["Prefer not to say", "Male", "Female", "Other"])

tab1, tab2, tab3 = st.tabs(["🧠 Ask Question", "📜 History", "🌐 Social Media Check"])

with tab1:
    question = st.text_input("Enter your medical question:")
    if st.button("Get Answer") and question:
        st.session_state.last_question = question
        demographics = f"For a {user_age}-year-old {user_gender.lower()}, " if user_age or user_gender != "Prefer not to say" else ""
        full_query = demographics + question
        with st.spinner("Generating response..."):
            answer, sources = answer_medical_question(full_query)
            risk_advisories = get_risk_snippets(question)
            severity = classify_severity(question)

        st.markdown(f"### 🚨 Severity Level: {severity}")
        st.markdown("### ✅ Answer")
        st.write(answer)

        if risk_advisories:
            st.markdown("### ⚠️ Proactive Health Advisory")
            for adv in risk_advisories:
                st.warning(adv)

        if sources:
            st.markdown("### 📚 Sources with Trust Scores")
            for title, link, snippet, score in sources:
                stars = "⭐" * int(score)
                st.markdown(f"- [{title}]({link}) ({stars})\n\n> {snippet}")

        st.session_state.history.append({
            "Question": question,
            "Answer": answer,
            "Sources": sources,
            "Severity": severity
        })

with tab2:
    st.markdown("### 📜 Your Session History")
    if not st.session_state.history:
        st.info("No questions asked yet.")
    else:
        for i, entry in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"**Q{i}: {entry['Question']}** ({entry['Severity']})")
            st.write(entry['Answer'])
            st.markdown("---")

with tab3:
    st.markdown("### 🌐 Social Media Medical Fact-Checking")
    sm_query = st.session_state.last_question

    if sm_query:
        with st.spinner("Retrieving and analyzing posts..."):
            sm_snippets = get_social_snippets(sm_query)
            risk_advisories = get_risk_snippets(sm_query)
            severity = classify_severity(sm_query)

        st.markdown(f"### 🚨 Severity Level: {severity}")

        if risk_advisories:
            st.markdown("### ⚠️ Proactive Health Advisory")
            for adv in risk_advisories:
                st.warning(adv)

        if not sm_snippets:
            st.warning("No relevant social media posts found.")
        else:
            st.markdown("### 🧾 Verified Posts from Social Media")
            for i, (title, link, snippet, score) in enumerate(sm_snippets, 1):
                stars = "⭐" * int(score)
                st.markdown(f"**Post {i}:** [{title}]({link}) ({stars})")
                st.markdown(f"> {snippet}")

                prompt = f"""
Below is a social media post snippet from a health-related discussion. Verify the medical information presented in it.
Use trusted guidelines (e.g., NHS, WHO, CDC) and specify what is correct or incorrect.

Post:
\"{snippet}\"

Respond clearly with bullet points:
- ✅ Valid claims
- ❌ Misinformation
- 🟢 Any advice or warning

Always end with: "Social media content may not be fully reliable. Consult a healthcare provider."
"""
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    fact_check = response.choices[0].message.content.strip()
                except Exception as e:
                    fact_check = f"Error verifying this post: {e}"

                st.markdown("**🔍 Fact-Check Result:**")
                st.info(fact_check)
                st.markdown("---")
    else:
        st.info("Ask a question in Tab 1 to populate social media analysis.")

# === Feedback Form ===
st.markdown("---")
st.markdown("### 💬 Leave Feedback")

creds = Credentials.from_service_account_info(GCP_SERVICE_ACCOUNT, scopes=[
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
])
gc = gspread.authorize(creds)
feedback_sheet = gc.open(GOOGLE_SHEET_NAME).sheet1

with st.form("feedback_form"):
    st.markdown("*(Optional)* Rate your experience and provide feedback.")
    rating = st.radio("How would you rate your experience?", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], index=4, horizontal=True)
    comments = st.text_area("Your Feedback")
    if st.form_submit_button("Submit Feedback"):
        feedback_sheet.append_row([rating, comments])
        st.success("✅ Thank you for your feedback!")

# === Footer ===
st.markdown("---")
st.caption("Developed by Doaa Al-Turkey")
