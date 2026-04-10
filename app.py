"""
JavaMind — PhD Research Demo
Title : Design and Implementation of Conversational AI in Academia
Guide : Dr. Satish Sankaye
Univ  : MGM University, Chhatrapati Sambhaji Nagar

Install:
    pip install streamlit transformers torch sentencepiece

Run:
    streamlit run java_qg_app.py
"""

import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JavaMind · PhD Research",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.main .block-container   { padding: 0 2rem 3rem; max-width: 1020px; }
section[data-testid="stSidebar"] { display: none; }

/* ══════════════════════════
   PHD HEADER BANNER
══════════════════════════ */
.phd-banner {
    background: #0a0f1e;
    border-bottom: 1px solid #1a2540;
    padding: 0.55rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 0;
}
.phd-left { display: flex; align-items: center; gap: 14px; }
.phd-logo {
    width: 36px; height: 36px; border-radius: 8px;
    background: linear-gradient(135deg, #1d4ed8, #7c3aed);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; font-weight: 700; color: #fff;
    flex-shrink: 0;
}
.phd-title-block {}
.phd-title-block .pt {
    font-size: 0.78rem; font-weight: 600; color: #e2e8f0;
    line-height: 1.3; letter-spacing: 0.01em;
}
.phd-title-block .ps {
    font-size: 0.66rem; color: #4a6080; margin-top: 1px;
}
.phd-right {
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
}
.phd-meta-item {
    display: flex; align-items: center; gap: 6px;
}
.phd-meta-icon {
    width: 22px; height: 22px; border-radius: 50%;
    background: #131d35; border: 1px solid #1e3a5f;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.65rem; color: #60a5fa; flex-shrink: 0;
}
.phd-meta-text .pmt { font-size: 0.62rem; color: #4a6080; line-height: 1.2; }
.phd-meta-text .pmv { font-size: 0.7rem; color: #94a3b8; font-weight: 500; line-height: 1.2; }
.phd-divider { width: 1px; height: 28px; background: #1a2540; }

/* ══════════════════════════
   LANDING PAGE
══════════════════════════ */
.landing-wrap {
    background: linear-gradient(160deg, #060d1a 0%, #0c1830 55%, #080f1e 100%);
    border-radius: 20px;
    margin: 1.5rem 0 1.8rem;
    overflow: hidden;
    border: 1px solid #1a2f50;
}
.landing-top {
    padding: 3rem 2.5rem 2rem;
    text-align: center;
    border-bottom: 1px solid #111d35;
}
.landing-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(29,78,216,0.15); border: 1px solid rgba(29,78,216,0.3);
    border-radius: 30px; padding: 5px 14px;
    font-size: 0.72rem; color: #93c5fd; margin-bottom: 1.4rem;
    font-weight: 500; letter-spacing: 0.04em;
}
.landing-top h1 {
    font-size: 2.6rem; font-weight: 700; color: #f0f6ff;
    letter-spacing: -0.04em; line-height: 1.15; margin-bottom: 0.8rem;
}
.landing-top h1 span { color: #60a5fa; }
.landing-top .tagline {
    font-size: 0.95rem; color: #4a6a96; max-width: 520px;
    margin: 0 auto 2.5rem; line-height: 1.7;
}
.role-cards { display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; }
.role-card {
    background: #0d1e35; border: 1px solid #1e3a5f;
    border-radius: 18px; padding: 2rem 1.8rem;
    width: 240px; text-align: center; cursor: pointer;
    transition: border-color .2s, transform .15s, background .2s;
    position: relative; overflow: hidden;
}
.role-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: transparent; transition: background .2s;
}
.role-card:hover { border-color: #3b82f6; transform: translateY(-4px); background: #0f2240; }
.role-card:hover::before { background: linear-gradient(90deg, #3b82f6, #7c3aed); }
.role-card .rc-icon  { font-size: 2.6rem; margin-bottom: 0.9rem; }
.role-card .rc-title { font-size: 1.05rem; font-weight: 700; color: #e2eaf8; margin-bottom: 0.5rem; }
.role-card .rc-desc  { font-size: 0.78rem; color: #3d5a7a; line-height: 1.65; }
.role-card .rc-tag {
    display: inline-block; margin-top: 1rem;
    background: rgba(37,99,235,0.12); border: 1px solid rgba(37,99,235,0.25);
    border-radius: 20px; padding: 3px 10px;
    font-size: 0.66rem; color: #60a5fa; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em;
}

.landing-bottom {
    display: flex; align-items: stretch;
    padding: 0;
}
.stat-panel {
    flex: 1; padding: 1.2rem 1.5rem;
    border-right: 1px solid #111d35;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center;
}
.stat-panel:last-child { border-right: none; }
.stat-panel .spv {
    font-size: 1.6rem; font-weight: 700; color: #60a5fa;
    font-family: 'JetBrains Mono', monospace; line-height: 1;
}
.stat-panel .spl { font-size: 0.68rem; color: #3d5a7a; margin-top: 4px;
                   text-transform: uppercase; letter-spacing: 0.06em; }

/* ══════════════════════════
   TOP BAR
══════════════════════════ */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.7rem 1.2rem;
    background: #fff; border: 1px solid #e5eaf5;
    border-radius: 14px; margin: 1.2rem 0 1.4rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.tb-left  { display: flex; align-items: center; gap: 12px; }
.tb-logo  { font-size: 1.05rem; font-weight: 700; color: #1e3a8a; }
.tb-role  {
    font-size: 0.68rem; font-weight: 600; padding: 3px 10px;
    border-radius: 20px; text-transform: uppercase; letter-spacing: 0.07em;
}
.role-teacher { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
.role-student { background: #fdf4ff; color: #7e22ce; border: 1px solid #e9d5ff; }
.tb-right { font-size: 0.7rem; color: #94a3b8; }

/* ══════════════════════════
   SHARED COMPONENTS
══════════════════════════ */
.sec-label {
    font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.09em; color: #94a3b8; margin-bottom: 0.5rem;
}
.panel {
    background: #fff; border: 1px solid #e5eaf5;
    border-radius: 16px; padding: 1.3rem 1.5rem; margin-bottom: 1.2rem;
}
.panel-title {
    font-size: 0.8rem; font-weight: 600; color: #1e3a8a;
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 1rem;
    padding-bottom: 0.6rem; border-bottom: 1px solid #eef2ff;
    display: flex; align-items: center; gap: 8px;
}
.stat-row { display: flex; gap: 10px; margin-bottom: 1.2rem; flex-wrap: wrap; }
.stat-b {
    flex: 1; min-width: 80px; text-align: center;
    background: #f8faff; border: 1px solid #e5eaf5;
    border-radius: 10px; padding: 0.8rem 0.5rem;
}
.stat-b .sv { font-size: 1.5rem; font-weight: 700; color: #1e3a8a;
              font-family: 'JetBrains Mono', monospace; }
.stat-b .sl { font-size: 0.65rem; color: #94a3b8; text-transform: uppercase;
              letter-spacing: 0.05em; margin-top: 3px; }
.info-chip {
    display: inline-flex; align-items: center; gap: 5px;
    background: #f0f9ff; border: 1px solid #bae6fd;
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.72rem; color: #0369a1; font-weight: 500;
    margin-bottom: 0.8rem;
}

/* ── Question cards ── */
.q-card {
    background: #f8faff; border: 1px solid #dde8f7;
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 10px;
    transition: border-color .15s;
}
.q-card:hover { border-color: #93c5fd; }
.q-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 50%;
    background: #1d4ed8; color: #fff;
    font-size: 0.7rem; font-weight: 700; margin-right: 10px; flex-shrink: 0;
}
.q-text { font-size: 0.95rem; color: #1e293b; line-height: 1.55; font-weight: 500; }
.q-meta { font-size: 0.7rem; color: #94a3b8; margin-top: 6px; padding-left: 36px; }
.bt-pill {
    display: inline-block; padding: 2px 9px; border-radius: 20px;
    font-size: 0.65rem; font-weight: 600; margin-right: 5px;
    text-transform: uppercase; letter-spacing: 0.04em;
}
.bt1{background:#eff6ff;color:#1d4ed8} .bt2{background:#f0fdf4;color:#15803d}
.bt3{background:#fefce8;color:#854d0e} .bt4{background:#faf5ff;color:#6d28d9}
.bt5{background:#fff1f2;color:#be123c} .bt6{background:#f0fdfa;color:#0f766e}

/* ══════════════════════════
   STUDENT CHAT
══════════════════════════ */
.s-hero {
    background: linear-gradient(135deg, #1a0533 0%, #2d1065 50%, #1a0533 100%);
    border-radius: 20px; padding: 2.8rem 2rem 2.2rem; text-align: center;
    margin-bottom: 1.5rem; border: 1px solid #4c1d95;
}
.s-hero .hero-icon { font-size: 2.8rem; margin-bottom: 0.8rem; }
.s-hero h2 {
    font-size: 1.8rem; font-weight: 700; color: #f5f0ff;
    letter-spacing: -0.03em; margin-bottom: 0.5rem;
}
.s-hero .sh { font-size: 0.85rem; color: #8060b0; margin-bottom: 1.5rem; line-height: 1.6; }
.s-hero .tip {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.3);
    border-radius: 30px; padding: 6px 16px; font-size: 0.76rem; color: #c4b5fd;
}
.ctx-strip {
    display: flex; align-items: center; gap: 10px;
    background: #fdf4ff; border: 1px solid #e9d5ff;
    border-radius: 12px; padding: 8px 14px; margin-bottom: 1rem;
    font-size: 0.8rem;
}
.ctx-strip .cs-label { color: #7e22ce; font-weight: 600; flex-shrink: 0; }
.ctx-strip .cs-preview { color: #6b21a8; font-style: italic;
                          overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.chat-outer {
    background: #fafbff; border: 1px solid #e8ecf8;
    border-radius: 18px; padding: 1.4rem 1.2rem 1rem;
    margin-bottom: 1rem; min-height: 300px;
}
.chat-messages { display: flex; flex-direction: column; gap: 16px; }
.msg-user { display: flex; justify-content: flex-end; }
.bubble-user {
    background: linear-gradient(135deg, #7c3aed, #6d28d9);
    color: #fff; border-radius: 18px 18px 4px 18px;
    padding: 0.85rem 1.1rem; max-width: 76%; font-size: 0.9rem; line-height: 1.65;
    box-shadow: 0 3px 10px rgba(109,40,217,0.22);
}
.msg-bot { display: flex; justify-content: flex-start; gap: 10px; align-items: flex-end; }
.bot-avatar {
    width: 34px; height: 34px; border-radius: 50%; flex-shrink: 0;
    background: linear-gradient(135deg, #1e3a8a, #1d4ed8);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; color: #fff; font-weight: 700; margin-bottom: 2px;
}
.bubble-bot {
    background: #fff; border: 1px solid #e2e8f5;
    border-radius: 18px 18px 18px 4px;
    padding: 0.9rem 1.1rem; max-width: 76%;
    font-size: 0.9rem; line-height: 1.75; color: #1e293b;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.bot-name {
    font-size: 0.63rem; font-weight: 700; color: #7c3aed;
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 5px;
}
.chips { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 0.75rem; }
.chip {
    background: #fdf4ff; border: 1px solid #e9d5ff;
    border-radius: 20px; padding: 5px 14px;
    font-size: 0.76rem; color: #7e22ce; cursor: pointer;
    transition: background .15s; white-space: nowrap; font-weight: 500;
}
.chip:hover { background: #ede9fe; }

/* ── Buttons ── */
div.stButton > button {
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 0.88rem !important;
    transition: all .2s !important;
}
.send-btn > button {
    background: linear-gradient(135deg,#7c3aed,#6d28d9) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important;
}
.send-btn > button:hover { opacity: 0.88 !important; }
.primary-btn > button {
    background: linear-gradient(135deg,#1d4ed8,#1e40af) !important;
    color: #fff !important; border: none !important;
}
.primary-btn > button:hover { opacity: 0.9 !important; }

/* ── Input ── */
textarea, input[type="text"] {
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── Footer ── */
.phd-footer {
    margin-top: 2rem;
    padding: 1rem 1.5rem;
    background: #f8faff;
    border: 1px solid #e5eaf5;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
}
.phd-footer .pf-left { font-size: 0.72rem; color: #64748b; }
.phd-footer .pf-right {
    font-size: 0.68rem; color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# ── PhD Header Banner ──────────────────────────────────────────────────────────
st.markdown("""
<div class="phd-banner">
  <div class="phd-left">
    <div class="phd-logo">J</div>
    <div class="phd-title-block">
      <div class="pt">Design and Implementation of Conversational AI in Academia</div>
      <div class="ps">PhD Research Prototype &nbsp;·&nbsp; DR.G.Y.P. College of CS and IT</div>
    </div>
  </div>
  <div class="phd-right">
    <div class="phd-meta-item">
      <div class="phd-meta-icon">G</div>
      <div class="phd-meta-text">
        <div class="pmt">Research Guide</div>
        <div class="pmv">Dr. Satish Sankaye</div>
      </div>
    </div>
    <div class="phd-divider"></div>
    <div class="phd-meta-item">
      <div class="phd-meta-icon">U</div>
      <div class="phd-meta-text">
        <div class="pmt">University</div>
        <div class="pmv">MGM University, CSN</div>
      </div>
    </div>
    <div class="phd-divider"></div>
    <div class="phd-meta-item">
      <div class="phd-meta-icon">M</div>
      <div class="phd-meta-text">
        <div class="pmt">Model</div>
        <div class="pmv">T5-Base · ROUGE-L 55.41</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path="./t5_java_qg_final"):
    tok = T5Tokenizer.from_pretrained(path)
    mdl = T5ForConditionalGeneration.from_pretrained(path)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(dev).eval()
    return tok, mdl, dev


def gen_questions(context, answer, tok, mdl, dev, n=3, beams=6, max_len=64):
    inp = "generate question: " + context.strip() + " answer: " + answer.strip()
    ids = tok(inp, max_length=512, truncation=True,
              return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        out = mdl.generate(
            ids, max_length=max_len,
            num_beams=max(beams, n), num_return_sequences=n,
            no_repeat_ngram_size=3, length_penalty=1.2, early_stopping=True,
        )
    return [tok.decode(o, skip_special_tokens=True).strip() for o in out]


# ── Span extraction ────────────────────────────────────────────────────────────
_BL = {
    "java","each","value","data","text","use","uses","used","method","methods",
    "object","provide","provides","meaning","means","store","stores","stored",
    "handling","manipulation","comparison","concatenation","sequence","api",
    "enclosed","changed","cannot","after","their","rich","double","single",
    "using","also","both","first","second","last","this","that","these",
    "those","which","where","when","what","who","how","why","has","have",
    "had","been","being","are","were","was","is","it","in","on","at","to",
    "for","of","and","or","but","not","with","from","into","about","before",
    "during","without","through","between","among","within","across","along",
    "character","characters","creation","following","certain","some","many",
    "any","all","more","most","less","few","several","various","other","same",
}

def _add(lst, seen, span, sent, kind):
    key = re.sub(r'\s+', ' ', span.strip().lower())
    words = key.split()
    if (key not in seen and 3 <= len(span.strip()) <= 80
            and 1 <= len(words) <= 8
            and not all(w in _BL for w in words)):
        seen.add(key)
        lst.append((span.strip(), sent.strip(), kind))

def extract_quality_spans(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    spans, seen = [], set()
    for sent in sentences:
        sent = sent.strip()
        if not sent: continue
        m = re.match(r'^(?:A[n]?\s+)([A-Za-z][\w\s]{1,30}?)\s+(?:in\s+\w+\s+)?(?:is|are)\b', sent, re.I)
        if m: _add(spans, seen, m.group(1).strip(), sent, "subject")
        m2 = re.search(r'\b(?:is|are)\s+(an?\s+)([\w\s\-]{4,60}?)(?:\s+used\s+to|\s+that|\s+which|[,.\n]|$)', sent, re.I)
        if m2:
            pred = m2.group(2).strip(' ,.')
            if len(pred.split()) <= 6: _add(spans, seen, pred, sent, "definition")
        m3 = re.search(r'\bused\s+to\s+([\w\s\-]{3,50}?)(?=[,.\n(]|$)', sent, re.I)
        if m3:
            action = m3.group(1).strip(' ,.')
            if len(action.split()) <= 6: _add(spans, seen, action, sent, "purpose")
        for phrase in re.findall(r'\b(\d+[\-\s]bit\s+\w+|[A-Z]{2,}[\-\d]+(?:\s+\w+)?|\w+[\-]\d+(?:\s+(?:encoding|format|bit|version|standard))?)\b', sent):
            _add(spans, seen, phrase.strip(), sent, "technical")
        for phrase in re.findall(r'(?:meaning|called|known as|referred to as|defined as|termed)\s+([A-Za-z][\w\s\-]{2,50}?)(?=[,.\n]|$)', sent, re.I):
            _add(spans, seen, phrase.strip(' ,.'), sent, "definition")
        for phrase in re.findall(r'\b(immutabl\w+|mutabl\w+|thread[\-\s]safe\w*|synchroniz\w+|platform[\-\s]independent|type[\-\s]safe|null[\-\s]safe|backward[\-\s]compatible|statically[\-\s]typed|strongly[\-\s]typed)\b', sent, re.I):
            _add(spans, seen, phrase.strip(), sent, "property")
        for phrase in re.findall(r'(?:stored?\s+using|uses?\s+|encoded?\s+(?:in|using|with))\s+([A-Za-z\d][\w\s\-]{2,40}?)(?=[,.\n(]|$)', sent, re.I):
            p = phrase.strip(' ,.()')
            if len(p.split()) <= 5: _add(spans, seen, p, sent, "technical")
        for phrase in re.findall(r'provides?\s+(?:a\s+)?(?:rich\s+)?([A-Za-z][\w\s]{2,30}?)\s+for\b', sent, re.I):
            _add(spans, seen, phrase.strip(), sent, "feature")
        for phrase in re.findall(r'"([^"]{2,40})"', sent):
            _add(spans, seen, phrase, sent, "quoted")
        for phrase in re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b', sent):
            if phrase.lower() not in _BL: _add(spans, seen, phrase, sent, "proper_noun")
    return spans

def jaccard(a, b):
    wa = set(re.findall(r'\b\w{3,}\b', a.lower()))
    wb = set(re.findall(r'\b\w{3,}\b', b.lower()))
    if not wa or not wb: return 0.0
    return len(wa & wb) / len(wa | wb)

def semantic_dedup(questions, threshold=0.55):
    kept = []
    for item in questions:
        if not any(jaccard(item["question"], k["question"]) >= threshold for k in kept):
            kept.append(item)
    return kept

_VAGUE = [
    r'^what is (?:the )?(?:literal )?name\b',
    r'^what (?:language|api|encoding|number) is used\b',
    r'^what is the number of',
    r'^what is (?:a |the )?(?:java|string|method|class|object)\s*\?',
    r'^what (?:does|do) (?:java|string|it|the)\b',
]

def is_good_question(question, answer, context):
    q = question.strip()
    if not q.endswith('?'):                               return False, "no ?"
    if len(q.split()) < 6:                               return False, "too short"
    q_lower = q.lower()
    starters = ("what","who","when","where","why","how","which",
                 "describe","explain","define","is","are","does","do",
                 "can","could","would","should","has","have","had","name")
    if not q_lower.startswith(starters):                 return False, "bad starter"
    if answer.lower() in q_lower and len(answer) > 4:   return False, "answer in q"
    q_w = set(re.findall(r'\b\w{4,}\b', q_lower))
    c_w = set(re.findall(r'\b\w{4,}\b', context.lower()))
    if len(q_w & c_w) < 2:                               return False, "low overlap"
    for pat in _VAGUE:
        if re.match(pat, q_lower):                       return False, "vague"
    return True, ""

def classify_bt(q):
    q = q.lower()
    if any(w in q for w in ["who","when","what is","what are","which","where","define","name","identify","list","recall"]):      return "BT1"
    if any(w in q for w in ["explain","describe","summarize","how does","what does","mean","purpose","interpret","classify"]):   return "BT2"
    if any(w in q for w in ["how would","demonstrate","use","apply","implement","calculate","show","solve","produce"]):          return "BT3"
    if any(w in q for w in ["compare","difference","why","analyse","distinguish","examine","break","differentiate","inspect"]): return "BT4"
    if any(w in q for w in ["evaluate","justify","assess","argue","defend","critique","recommend","judge","appraise"]):          return "BT5"
    if any(w in q for w in ["design","create","construct","develop","formulate","propose","build","compose","plan"]):            return "BT6"
    return "BT2"

def answer_from_context(question, context):
    q_lower   = question.lower()
    sentences = re.split(r'(?<=[.!?])\s+', context.strip())
    q_words   = set(re.findall(r'\b\w{3,}\b', q_lower))
    if any(w in q_lower for w in ["main topic","about","summary","summarize","overview"]):
        first = sentences[0] if sentences else context
        spans = extract_quality_spans(context)
        terms = ", ".join(f"<b>{s[0]}</b>" for s in spans[:5]) if spans else "various concepts"
        return f"This content covers:<br><br><i>{first}</i><br><br>Key concepts: {terms}"
    if any(w in q_lower for w in ["simple","easy","layman","beginner","explain simply"]):
        return f"In simple terms:<br><br><i>{' '.join(sentences[:2])}</i>"
    if any(w in q_lower for w in ["key term","technical term","important word","keyword","concept"]):
        spans = extract_quality_spans(context)
        if spans:
            items = "<br>".join(f"• <b>{s[0]}</b> — <i>{s[2]}</i>" for s in spans[:8])
            return f"Key concepts:<br><br>{items}"
        return "No specific technical terms were identified."
    if any(w in q_lower for w in ["example","instance","such as","for example"]):
        for sent in sentences:
            if any(w in sent.lower() for w in ["example","such as","like","e.g","instance"]):
                return f"From the content:<br><br><i>{sent}</i>"
        return f"Most relevant excerpt:<br><br><i>{max(sentences, key=len)}</i>"
    if any(w in q_lower for w in ["difference","compare","vs","versus","distinguish"]):
        contrast = [s for s in sentences if any(
            w in s.lower() for w in ["while","whereas","however","unlike","but","although"])]
        if contrast:
            return "Contrast found:<br><br>" + "<br><br>".join(f"<i>{s}</i>" for s in contrast[:2])
    if any(w in q_lower for w in ["how many","count","number of"]):
        nums = re.findall(r'\b\d+(?:\.\d+)?\b', context)
        if nums: return f"Numbers mentioned: <b>{', '.join(sorted(set(nums)))}</b>"
    if any(w in q_lower for w in ["why","reason","purpose","benefit","advantage"]):
        for sent in sentences:
            if any(w in sent.lower() for w in ["because","since","therefore","thus","reason","purpose","allow","enable"]):
                return f"<i>{sent}</i>"
    scored = sorted([(len(q_words & set(re.findall(r'\b\w{3,}\b', s.lower()))), s)
                     for s in sentences], reverse=True)
    if scored and scored[0][0] > 0:
        best  = scored[0][1]
        spans = extract_quality_spans(best)
        if spans:
            best_span = max(spans, key=lambda s: len(
                set(re.findall(r'\b\w+\b', s[0].lower())) & q_words), default=None)
            if best_span and len(best_span[0]) > 3:
                return f"<b>{best_span[0]}</b><br><br><i>From context:</i> {best}"
        return f"<i>{best}</i>"
    return "I couldn't find a clear answer in the provided content. Try rephrasing your question."


# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {"role": None, "chat": [], "ctx": "", "ctx_set": False}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
#  LANDING
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.role is None:
    st.markdown("""
    <div class="landing-wrap">
      <div class="landing-top">
        <div class="landing-badge">☕ &nbsp; AI-Powered Java Learning System</div>
        <h1>JavaMind<br><span>Question Generation &amp; Tutoring</span></h1>
        <p class="tagline">
          A PhD research prototype that uses a fine-tuned T5-Base transformer to automatically
          generate pedagogically-grounded questions from Java content, aligned with
          Bloom's Taxonomy cognitive levels BT1–BT6.
        </p>
        <div class="role-cards">
          <div class="role-card">
            <div class="rc-icon">👩‍🏫</div>
            <div class="rc-title">Teacher Portal</div>
            <div class="rc-desc">
              Paste any Java paragraph — the AI extracts key concepts and
              automatically generates unique, quality-filtered questions
              tagged with Bloom's Taxonomy levels.
            </div>
            <div class="rc-tag">Question Generation</div>
          </div>
          <div class="role-card">
            <div class="rc-icon">👨‍🎓</div>
            <div class="rc-title">Student Portal</div>
            <div class="rc-desc">
              Paste your Java study notes and chat with the AI —
              ask anything about the content and get instant,
              context-grounded answers.
            </div>
            <div class="rc-tag">Conversational AI</div>
          </div>
        </div>
      </div>
      <div class="landing-bottom">
        <div class="stat-panel"><div class="spv">1,993</div><div class="spl">Training samples</div></div>
        <div class="stat-panel"><div class="spv">135</div><div class="spl">Java topics</div></div>
        <div class="stat-panel"><div class="spv">BT1–6</div><div class="spl">Bloom's levels</div></div>
        <div class="stat-panel"><div class="spv">55.41</div><div class="spl">ROUGE-L score</div></div>
        <div class="stat-panel"><div class="spv">222M</div><div class="spl">Model parameters</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 0.3, 1])
    with c1:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("👩‍🏫  Enter as Teacher", use_container_width=True):
            st.session_state.role = "teacher"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="send-btn">', unsafe_allow_html=True)
        if st.button("👨‍🎓  Enter as Student", use_container_width=True):
            st.session_state.role = "student"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="phd-footer">
      <div class="pf-left">
        <b>Research Title:</b> Design and Implementation of Conversational AI in Academia &nbsp;·&nbsp;
        <b>Guide:</b> Dr. Satish Sankaye &nbsp;·&nbsp;
        <b>University:</b> MGM University, Chhatrapati Sambhaji Nagar
      </div>
      <div class="pf-right">T5-Base &nbsp;|&nbsp; HuggingFace Transformers 4.46 &nbsp;|&nbsp; Streamlit</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Top bar (shared) ───────────────────────────────────────────────────────────
rl = st.session_state.role
st.markdown(f"""
<div class="topbar">
  <div class="tb-left">
    <span style="font-size:1.2rem">☕</span>
    <span class="tb-logo">JavaMind</span>
    <span class="tb-role role-{'teacher' if rl=='teacher' else 'student'}">
      {'👩‍🏫 Teacher Portal' if rl=='teacher' else '👨‍🎓 Student Portal'}
    </span>
    <span style="font-size:0.68rem;color:#cbd5e1;padding:2px 10px;background:#f1f5f9;border-radius:20px;">
      Design and Implementation of Conversational AI in Academia
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

with st.columns([5, 1])[1]:
    if st.button("Switch role"):
        st.session_state.update({"role": None, "chat": [], "ctx": "", "ctx_set": False})
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  TEACHER PORTAL
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.role == "teacher":

    try:
        tok, mdl, dev = load_model()
        device_label = "GPU (CUDA)" if str(dev) == "cuda" else "CPU"
        st.markdown(f'<div class="info-chip">● Model ready &nbsp;·&nbsp; {device_label} &nbsp;·&nbsp; T5-Base 222M params</div>',
                    unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Model load failed: {e}"); st.stop()

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📄 Step 1 &nbsp; Paste Java content</div>', unsafe_allow_html=True)
    context = st.text_area("ctx", label_visibility="collapsed",
        placeholder=(
            "Paste any Java paragraph here — the AI will automatically extract "
            "all key concepts and generate unique, quality-filtered questions.\n\n"
            "Example:\n"
            "A String in Java is an object used to store a sequence of characters "
            "enclosed in double quotes. It uses UTF-16 encoding and provides methods "
            "for handling text data. Strings are immutable, meaning their value "
            "cannot be changed after creation. Java provides a rich API for "
            "manipulation, comparison, and concatenation of strings."
        ),
        height=160)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">⚙️ Step 2 &nbsp; Configure generation</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        beams    = st.slider("Generation quality (beam size)", 4, 10, 6,
                             help="Higher = better quality questions, slower")
    with c2:
        max_span = st.slider("Max concepts to extract", 3, 20, 10,
                             help="How many answer spans to generate questions for")
    with c3:
        n_gen    = st.slider("Questions per concept", 1, 3, 2,
                             help="Duplicates are removed automatically")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
    generate = st.button("Generate Questions", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if generate:
        if not context.strip():
            st.warning("Please paste a Java paragraph first.")
        else:
            spans = extract_quality_spans(context)[:max_span]
            if not spans:
                st.error("No meaningful concepts found. Try a richer technical paragraph.")
            else:
                prog   = st.progress(0, text="Analysing concepts...")
                all_qs = []
                for i, (span, source_sent, kind) in enumerate(spans):
                    prog.progress((i + 1) / len(spans),
                                  text=f"Generating for concept: *{span}*")
                    try:
                        for q in gen_questions(source_sent, span, tok, mdl, dev,
                                               n=n_gen, beams=beams):
                            ok, _ = is_good_question(q, span, context)
                            if ok:
                                all_qs.append({"question": q, "answer": span,
                                               "bt": classify_bt(q), "kind": kind})
                    except Exception:
                        continue
                prog.empty()

                # Dedup stage 1: prefix
                prefix_seen, after_prefix = set(), []
                for item in all_qs:
                    key = re.sub(r'\W+', '', item["question"].lower())[:55]
                    if key not in prefix_seen:
                        prefix_seen.add(key); after_prefix.append(item)

                # Dedup stage 2: Jaccard semantic
                bt_order = {"BT1":0,"BT2":1,"BT3":2,"BT4":3,"BT5":4,"BT6":5}
                after_prefix.sort(key=lambda x: bt_order.get(x["bt"], 9))
                unique = semantic_dedup(after_prefix, threshold=0.55)

                if not unique:
                    st.warning("No high-quality questions generated. "
                               "Try increasing beam size or a richer paragraph.")
                else:
                    bt_counts   = {}
                    for item in unique:
                        bt_counts[item["bt"]] = bt_counts.get(item["bt"], 0) + 1
                    dup_removed = len(all_qs) - len(unique)

                    st.markdown(f"""
                    <div class="stat-row">
                      <div class="stat-b"><div class="sv">{len(unique)}</div><div class="sl">Unique questions</div></div>
                      <div class="stat-b"><div class="sv">{len(spans)}</div><div class="sl">Concepts found</div></div>
                      <div class="stat-b"><div class="sv">{dup_removed}</div><div class="sl">Duplicates removed</div></div>
                      {"".join(f'<div class="stat-b"><div class="sv">{v}</div><div class="sl">{k}</div></div>'
                               for k, v in sorted(bt_counts.items()))}
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f'<div class="sec-label">{len(unique)} unique quality questions — sorted by Bloom\'s Taxonomy level</div>',
                                unsafe_allow_html=True)

                    for i, item in enumerate(unique, 1):
                        st.markdown(f"""
                        <div class="q-card">
                          <div style="display:flex;align-items:flex-start">
                            <span class="q-num">{i}</span>
                            <span class="q-text">{item['question']}</span>
                          </div>
                          <div class="q-meta">
                            <span class="bt-pill {item['bt'].lower()}">{item['bt']}</span>
                            Answer: <b>{item['answer']}</b>
                            &nbsp;·&nbsp; Concept type: {item['kind']}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                    lines = [
                        "JavaMind — Auto-Generated Questions",
                        "PhD Research: Design and Implementation of Conversational AI in Academia",
                        "Guide: Dr. Satish Sankaye | MGM University, Chhatrapati Sambhaji Nagar",
                        "", f"Context:\n{context}", "", "─" * 60, ""
                    ]
                    for i, item in enumerate(unique, 1):
                        lines.append(f"Q{i} [{item['bt']}]: {item['question']}\nAnswer: {item['answer']}\n")
                    st.download_button(
                        "Download questions (.txt)",
                        data="\n".join(lines),
                        file_name="java_questions.txt",
                        mime="text/plain",
                    )

    st.markdown("""
    <div class="phd-footer" style="margin-top:1.5rem">
      <div class="pf-left">
        <b>How it works:</b> T5-Base fine-tuned on Java-2000 dataset (1,993 QA pairs) ·
        Rule-based span extraction · Two-stage Jaccard deduplication · 5-criterion quality filter
      </div>
      <div class="pf-right">ROUGE-L: 55.41 &nbsp;|&nbsp; Epoch: 7.99 &nbsp;|&nbsp; Train loss: 0.58</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  STUDENT PORTAL
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.role == "student":

    if not st.session_state.ctx_set:
        st.markdown("""
        <div class="s-hero">
          <div class="hero-icon">🧠</div>
          <h2>Your Personal Java Tutor</h2>
          <p class="sh">
            Paste any Java study content — textbook paragraphs, lecture notes, or documentation.
            Then ask anything and get instant answers powered by AI.
          </p>
          <div class="tip">
            <span>📌</span>
            <span>Works with any Java topic — OOP, Collections, JVM, Threads, and more</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📚 Paste your Java study content</div>',
                    unsafe_allow_html=True)
        ctx_in = st.text_area("content", label_visibility="collapsed",
            placeholder=(
                "Paste your Java notes, textbook paragraph, or documentation here...\n\n"
                "Example:\n"
                "A String in Java is an object used to store a sequence of characters "
                "enclosed in double quotes. It uses UTF-16 encoding and provides methods "
                "for handling text data. Strings are immutable, meaning their value "
                "cannot be changed after creation..."
            ),
            height=200)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="send-btn">', unsafe_allow_html=True)
        if st.button("Start Chatting  →", use_container_width=True):
            if ctx_in.strip():
                st.session_state.ctx     = ctx_in.strip()
                st.session_state.ctx_set = True
                wc = len(ctx_in.split())
                st.session_state.chat = [{"role": "bot", "content":
                    f"Hello! I've read your Java content (<b>{wc} words</b>). "
                    f"Ask me anything — definitions, explanations, comparisons, "
                    f"examples, or just type a topic and I'll explain it from your notes!"}]
                st.rerun()
            else:
                st.warning("Please paste some Java content first.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    # ── Chat interface ─────────────────────────────────────────────────────────
    preview = st.session_state.ctx[:70] + "..." if len(st.session_state.ctx) > 70 else st.session_state.ctx
    st.markdown(f"""
    <div class="ctx-strip">
      <span class="cs-label">📄 Content loaded:</span>
      <span class="cs-preview">{preview}</span>
    </div>
    """, unsafe_allow_html=True)

    col_chg, _ = st.columns([1, 5])
    with col_chg:
        if st.button("Change content"):
            st.session_state.ctx_set = False
            st.session_state.chat    = []
            st.session_state.ctx     = ""
            st.rerun()

    # Chat bubbles
    chat_html = '<div class="chat-outer"><div class="chat-messages">'
    for msg in st.session_state.chat:
        if msg["role"] == "user":
            chat_html += f'<div class="msg-user"><div class="bubble-user">{msg["content"]}</div></div>'
        else:
            content = msg["content"]
            content = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', content)
            content = re.sub(r'\*(.+?)\*',     r'<i>\1</i>', content)
            content = content.replace("\n", "<br>")
            chat_html += f"""
            <div class="msg-bot">
              <div class="bot-avatar">J</div>
              <div class="bubble-bot">
                <div class="bot-name">JavaMind AI</div>
                {content}
              </div>
            </div>"""
    chat_html += '</div></div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # Quick chips
    chips = [
        ("🧩 Main topic",           "What is the main topic of this content?"),
        ("🔑 Key concepts",         "What are the key technical terms?"),
        ("📖 Explain simply",       "Can you explain this in simple terms?"),
        ("💡 Give an example",      "Can you give me an example from this content?"),
        ("⚡ Why important?",       "Why is this concept important in Java?"),
    ]
    chip_cols = st.columns(len(chips))
    quick_q   = None
    for col, (label, prompt) in zip(chip_cols, chips):
        with col:
            if st.button(label, key=f"chip_{label}", use_container_width=True):
                quick_q = prompt

    # Input + send
    st.markdown('<div style="margin-top:0.6rem"></div>', unsafe_allow_html=True)
    inp_col, btn_col = st.columns([5, 1])
    with inp_col:
        user_q = st.text_input("msg", label_visibility="collapsed",
            placeholder="Type your question about the Java content...",
            key="student_input")
    with btn_col:
        st.markdown('<div class="send-btn">', unsafe_allow_html=True)
        send = st.button("Send ➤", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    question = quick_q or (user_q.strip() if send and user_q.strip() else None)
    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        ans = answer_from_context(question, st.session_state.ctx)
        st.session_state.chat.append({"role": "bot", "content": ans})
        st.rerun()

    if len(st.session_state.chat) > 1:
        st.markdown('<div style="margin-top:0.5rem"></div>', unsafe_allow_html=True)
        if st.button("🗑 Clear chat"):
            st.session_state.chat = [{"role": "bot",
                "content": "Chat cleared! What would you like to know?"}]
            st.rerun()

    st.markdown("""
    <div class="phd-footer" style="margin-top:1.5rem">
      <div class="pf-left">
        <b>JavaMind</b> — PhD Research Prototype &nbsp;·&nbsp;
        Design and Implementation of Conversational AI in Academia &nbsp;·&nbsp;
        Guide: Dr. Satish Sankaye &nbsp;·&nbsp; MGM University, Chhatrapati Sambhaji Nagar
      </div>
      <div class="pf-right">Conversational AI Module</div>
    </div>
    """, unsafe_allow_html=True)
