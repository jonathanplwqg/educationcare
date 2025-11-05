# streamlit_english_test.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

st.set_page_config(page_title="EducationCare • English Test", layout="centered")
st.title("🟢 EducationCare — Mini English Proficiency Test")
st.write("Answer the questions below. We'll score you and suggest what to improve.")

# 1) try to load trained model (optional)
model = None
if os.path.exists("english_proficiency_model.pkl"):
    try:
        model = joblib.load("english_proficiency_model.pkl")
        st.success("Model loaded.")
    except Exception as e:
        st.warning(f"Could not load model: {e}")
else:
    st.info("No model found — will show rule-based recommendations only.")

# ---------------------------------------------------------------------
# 2) DEFINE QUESTIONS (you can add more later)
# ---------------------------------------------------------------------
# Vocabulary: 4 sample questions → we will scale to /10 later
vocab_questions = [
    {
        "q": "Choose the best meaning of the word 'rapid'.",
        "options": ["slow", "fast", "heavy", "short"],
        "answer": "fast",
    },
    {
        "q": "Choose the synonym of 'begin'.",
        "options": ["start", "stop", "close", "late"],
        "answer": "start",
    },
    {
        "q": "Choose the opposite of 'difficult'.",
        "options": ["easy", "hard", "strong", "quick"],
        "answer": "easy",
    },
    {
        "q": "Choose the correct word: 'She bought a _____ of bread.'",
        "options": ["loaf", "leaf", "piece", "glass"],
        "answer": "loaf",
    },
]

# Grammar: 4 sample questions → scale to /8
grammar_questions = [
    {
        "q": "He ____ to school every day.",
        "options": ["go", "goes", "going", "gone"],
        "answer": "goes",
    },
    {
        "q": "They ____ dinner right now.",
        "options": ["are cooking", "cook", "cooks", "cooked"],
        "answer": "are cooking",
    },
    {
        "q": "I have ____ this movie before.",
        "options": ["see", "saw", "seen", "seeing"],
        "answer": "seen",
    },
    {
        "q": "She didn’t ____ to the party.",
        "options": ["go", "goes", "going", "gone"],
        "answer": "go",
    },
]

# Reading: 3 questions → scale to /6
reading_questions = [
    {
        "text": "Liam was late to class because the bus broke down on the way.",
        "q": "Why was Liam late?",
        "options": [
            "He overslept",
            "The bus broke down",
            "He forgot his homework",
            "It was a holiday"
        ],
        "answer": "The bus broke down",
    },
    {
        "text": "Maria drinks coffee every morning, but today she chose tea.",
        "q": "What did Maria drink today?",
        "options": ["Coffee", "Tea", "Juice", "Milk"],
        "answer": "Tea",
    },
    {
        "text": "The restaurant was full, so they had to wait for a table.",
        "q": "What was the problem?",
        "options": ["The food was bad", "They were late", "It was full", "It was closed"],
        "answer": "It was full",
    },
]

# Writing mechanics: 2 questions → scale to /5
writing_questions = [
    {
        "q": "Choose the correct sentence.",
        "options": [
            "She don't like apples.",
            "She doesn't like apples.",
            "She doesn't likes apples.",
            "She not like apples."
        ],
        "answer": "She doesn't like apples.",
    },
    {
        "q": "Choose the sentence with correct punctuation.",
        "options": [
            "Yesterday I went to London Paris and Rome.",
            "Yesterday, I went to London, Paris, and Rome.",
            "Yesterday I went, to London, Paris and Rome.",
            "Yesterday I went to London, Paris and, Rome.",
        ],
        "answer": "Yesterday, I went to London, Paris, and Rome.",
    },
]

# ---------------------------------------------------------------------
# 3) BUILD THE TEST UI
# ---------------------------------------------------------------------
with st.form("english_test_form"):
    st.subheader("🟣 Part 1 — Vocabulary")
    vocab_answers = []
    for i, item in enumerate(vocab_questions, start=1):
        ans = st.radio(f"{i}. {item['q']}", item["options"], key=f"vocab_{i}")
        vocab_answers.append(ans)

    st.subheader("🟣 Part 2 — Grammar")
    grammar_answers = []
    for i, item in enumerate(grammar_questions, start=1):
        ans = st.radio(f"{i}. {item['q']}", item["options"], key=f"grammar_{i}")
        grammar_answers.append(ans)

    st.subheader("🟣 Part 3 — Reading")
    reading_answers = []
    for i, item in enumerate(reading_questions, start=1):
        st.write(f"**Text:** {item['text']}")
        ans = st.radio(f"{i}. {item['q']}", item["options"], key=f"reading_{i}")
        reading_answers.append(ans)

    st.subheader("🟣 Part 4 — Writing / Mechanics")
    writing_answers = []
    for i, item in enumerate(writing_questions, start=1):
        ans = st.radio(f"{i}. {item['q']}", item["options"], key=f"writing_{i}")
        writing_answers.append(ans)

    submitted = st.form_submit_button("✅ Submit and get feedback")

# ---------------------------------------------------------------------
# 4) WHEN SUBMITTED → SCORE
# ---------------------------------------------------------------------
def score_section(answers, questions):
    score = 0
    for ans, q in zip(answers, questions):
        if ans == q["answer"]:
            score += 1
    return score

if submitted:
    # raw scores
    vocab_raw = score_section(vocab_answers, vocab_questions)          # out of 4
    grammar_raw = score_section(grammar_answers, grammar_questions)    # out of 4
    reading_raw = score_section(reading_answers, reading_questions)    # out of 3
    writing_raw = score_section(writing_answers, writing_questions)    # out of 2

    # scale to your original ranges
    vocab_score_10 = round(vocab_raw * (10 / len(vocab_questions)))       # scale 0–10
    grammar_score_8 = round(grammar_raw * (8 / len(grammar_questions)))   # scale 0–8
    reading_inference_6 = round(reading_raw * (6 / len(reading_questions)))  # 0–6
    writing_mechanics_5 = round(writing_raw * (5 / len(writing_questions)))  # 0–5

    st.write("### ✔️ Your raw results")
    st.write(f"Vocabulary: {vocab_score_10}/10")
    st.write(f"Grammar: {grammar_score_8}/8")
    st.write(f"Reading: {reading_inference_6}/6")
    st.write(f"Writing: {writing_mechanics_5}/5")

    # normalize to 0–1 (same as your classifier)
    user_scores = {
        "Vocabulary": vocab_score_10 / 10,
        "Grammar": grammar_score_8 / 8,
        "Reading": reading_inference_6 / 6,
        "Writing": writing_mechanics_5 / 5,
    }

    # model-facing names (must match training)
    model_scores = {
        "vocab_norm":   user_scores["Vocabulary"],
        "grammar_norm": user_scores["Grammar"],
        "reading_norm": user_scores["Reading"],
        "writing_norm": user_scores["Writing"],
    }

    # show levels
    def level_from_score(s):
        if s < 0.60:
            return "Needs Work"
        elif s < 0.80:
            return "Adequate"
        return "Strong"

    st.subheader("📊 Skill levels")
    levels_df = pd.DataFrame({
        "Skill": list(user_scores.keys()),
        "Score (0–1)": [round(v, 2) for v in user_scores.values()],
        "Level": [level_from_score(v) for v in user_scores.values()]
    })
    st.dataframe(levels_df, hide_index=True, use_container_width=True)

    # classifier prediction
    if model is not None:
        X_one = pd.DataFrame([model_scores])
        pred_class = int(model.predict(X_one)[0])
        label_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
        st.success(f"🧠 Overall English Proficiency (model): **{label_map[pred_class]}**")
    else:
        st.info("No model loaded — skipping overall proficiency.")

    # recommendations (rule-based)
    RECS = {
        "Vocabulary": {
            "Needs Work": "Do 15–20 vocab items/day + review yesterday.",
            "Adequate": "Keep 10 items/day and start B1 words.",
            "Strong": "Move to B2/C1 lists and revise weekly."
        },
        "Grammar": {
            "Needs Work": "Do 10 gap-fill/cloze tasks on tenses/S–V.",
            "Adequate": "Mix 5–8 grammar items/day.",
            "Strong": "Work on complex sentences and connectors."
        },
        "Reading": {
            "Needs Work": "Read 1 short A2/B1 text + 3 questions/day.",
            "Adequate": "Alternate inference and cloze tasks.",
            "Strong": "Read authentic articles and summarize."
        },
        "Writing": {
            "Needs Work": "Fix 5 sentences/day (spelling/punctuation).",
            "Adequate": "Write 1 short paragraph/day.",
            "Strong": "Write 150–200 words/week with cohesion focus."
        },
    }

    st.subheader("🧭 Personalized recommendations")
    # order by weakest
    ordered = sorted(user_scores.items(), key=lambda x: x[1])
    for skill, score in ordered[:3]:
        lvl = level_from_score(score)
        st.write(f"**{skill}** — {lvl} (score {score:.2f})")
        st.write(RECS[skill][lvl])

