# =========================================
# ULTRA JAHANNAM Interactive v1 🔥
# Hybrid AI + Semantic + Rules + Attention
# Streamlit GUI
# =========================================

import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# =========================================
# 1️⃣ DEVICE & MODELS
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(device)

# Semantic
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# =========================================
# 2️⃣ ULTRA CLASSIFIER
# =========================================
class UltraClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = bert
        self.attention = nn.Linear(768, 1)
        self.classifier = nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        weights = torch.softmax(self.attention(hidden), dim=1)
        context = torch.sum(weights * hidden, dim=1)
        logits = self.classifier(context)
        return logits, weights

model = UltraClassifier().to(device)

# =========================================
# 3️⃣ KNOWLEDGE ANCHORS
# =========================================
anchors = {
    "Hardware": ["electronic circuit design", "voltage and current analysis", "pcb and embedded systems"],
    "Software": ["programming and debugging", "software development and code", "algorithms and data structures"]
}
anchor_embeddings = {
    k: torch.tensor(semantic_model.encode(v, convert_to_tensor=True), device=device)
    for k,v in anchors.items()
}

# =========================================
# 4️⃣ SEMANTIC SCORING
# =========================================
def semantic_score(text):
    text_vec = torch.tensor(semantic_model.encode([text], convert_to_tensor=True), device=device)
    scores = {}
    for category, vecs in anchor_embeddings.items():
        sims = cosine_similarity(vecs.cpu().numpy(), text_vec.cpu().numpy().reshape(1, -1)).flatten()
        scores[category] = float(np.mean(sims))
    return scores

# =========================================
# 5️⃣ RULE ENGINE
# =========================================
HARDWARE = ["circuit", "voltage", "pcb", "sensor", "mouse"]
SOFTWARE = ["python", "bug", "code", "api", "software"]

def rule_engine(text):
    text = text.lower()
    hw = sum(word in text for word in HARDWARE)
    sw = sum(word in text for word in SOFTWARE)
    return {"Hardware": hw, "Software": sw}

# =========================================
# 6️⃣ PREDICTION FUNCTION
# =========================================
def predict(text, fusion_weights=(0.5,0.3,0.2)):
    w_bert, w_sem, w_rule = fusion_weights
    model.eval()
    
    # BERT
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits, attn = model(enc["input_ids"], enc["attention_mask"])
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    bert_scores = {"Hardware": probs[0], "Software": probs[1], "Mixed": probs[2]}

    # Semantic
    sem_scores = semantic_score(text)

    # Rule
    rule_scores = rule_engine(text)

    # Fusion
    final_scores = {}
    for key in ["Hardware", "Software"]:
        final_scores[key] = w_bert*bert_scores.get(key,0) + w_sem*sem_scores.get(key,0) + w_rule*rule_scores.get(key,0)
    final_scores["Mixed"] = bert_scores["Mixed"]

    prediction = max(final_scores, key=final_scores.get)

    # Top keywords
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    attn_weights = attn[0].squeeze().cpu().numpy()
    mask_tokens = [t for t in tokens if t not in ["[PAD]", "[CLS]", "[SEP]"]]
    mask_weights = attn_weights[1:len(mask_tokens)+1]
    important = sorted(zip(mask_tokens, mask_weights), key=lambda x:x[1], reverse=True)[:5]

    return {
        "prediction": prediction,
        "scores": final_scores,
        "top_keywords": [w for w,_ in important]
    }

# =========================================
# 7️⃣ STREAMLIT GUI
# =========================================
st.title("🔥 Noor AI-Hussein")
st.write("Type your problem below and the system will classify it as Hardware, Software, or Mixed.")

user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a text to predict.")
    else:
        result = predict(user_input)
        st.success(f"Prediction: **{result['prediction']}**")
        st.write("Top Keywords:", result["top_keywords"])
        st.write("Scores:", result["scores"])
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image("شعار الفارابي.jpg", width=200)
import streamlit as st
import streamlit as st

st.markdown(
    """
    <style>
    /* 1. اللون الأحمر المغبر الأساسي */
    .stApp {
        background-color: #8B3A3A;
        overflow: hidden;
    }

    /* 2. حركة النجوم (تومض وتتحرك يمين ويسار) */
    @keyframes starMovement {
        0% { opacity: 0.2; transform: translate(0, 0); }
        50% { opacity: 1; transform: translate(20px, 10px); }
        100% { opacity: 0.2; transform: translate(0, 0); }
    }

    .stApp::before {
        content: " ";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            radial-gradient(1.5px 1.5px at 15% 25%, white, transparent),
            radial-gradient(2px 2px at 40% 70%, white, transparent),
            radial-gradient(1px 1px at 80% 30%, white, transparent),
            radial-gradient(1.5px 1.5px at 20% 85%, white, transparent),
            radial-gradient(2px 2px at 60% 15%, white, transparent);
        animation: starMovement 4s infinite ease-in-out;
        z-index: 0;
    }

    /* 3. الرموز الهندسية والتقنية (تتحرك وتطوف بالخلفية) */
    @keyframes floatIcons {
        0% { transform: translateY(0) translateX(0) rotate(0deg); opacity: 0.1; }
        50% { transform: translateY(-20px) translateX(15px) rotate(5deg); opacity: 0.3; }
        100% { transform: translateY(0) translateX(0) rotate(0deg); opacity: 0.1; }
    }

    .tech-icon {
        position: fixed;
        color: rgba(255, 255, 255, 0.4);
        font-size: 35px;
        z-index: 0;
        pointer-events: none;
        animation: floatIcons 6s infinite ease-in-out;
    }
    </style>

    <div class="tech-icon" style="top: 15%; left: 10%;">💻</div>
    <div class="tech-icon" style="top: 25%; right: 15%; animation-delay: 1s;">⚙️</div>
    <div class="tech-icon" style="bottom: 20%; left: 20%; animation-delay: 2s;">🔒</div>
    <div class="tech-icon" style="top: 50%; left: 5%; animation-delay: 3s;">{ }</div>
    <div class="tech-icon" style="bottom: 10%; right: 10%; animation-delay: 1.5s;">📟</div>
    <div class="tech-icon" style="top: 70%; right: 30%; animation-delay: 2.5s;">ENG</div>
    <div class="tech-icon" style="top: 10%; right: 40%; animation-delay: 4s;">💻</div>
    <div class="tech-icon" style="bottom: 40%; right: 5%; animation-delay: 0.5s;">🔒</div>
    """,
    unsafe_allow_html=True
)