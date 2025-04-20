# AI Company Report Agent – Streamlit App with Multimodal AI and Enhanced UI

from __future__ import annotations
import os, io, re, base64, shutil, requests
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
from PIL import Image
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as rl_canvas

from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBED_MODEL = "models/embedding-001"
VLLM_MODEL = "models/gemini-1.5-pro-latest"

vision_llm = ChatGoogleGenerativeAI(model=VLLM_MODEL, google_api_key=GOOGLE_API_KEY)
text_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

st.set_page_config(page_title="AI Company Report Agent", layout="wide")
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stButton>button, .stDownloadButton>button {
        border-radius: 8px;
        font-size: 16px;
        padding: 8px 20px;
    }
</style>
<h1 style='text-align: center;'>Agent 10K: Multimodal Annual Report Analyst</h1><hr>
""", unsafe_allow_html=True)

# File Upload UI
with st.sidebar:
    st.header("📁 Upload Report(s)")
    pdfs = st.file_uploader("Upload one or more Annual Reports (PDF)", type=["pdf"], accept_multiple_files=True)
    st.info("Supports multimodal analysis (text + charts/images)", icon="ℹ️")

# Core Functions
def _extract_pdf_contents(files: List[io.BytesIO]) -> Tuple[List[str], List[bytes]]:
    texts, images = [], []
    for f in files:
        with fitz.open(stream=f.read(), filetype="pdf") as doc:
            for page in doc:
                texts.append(page.get_text("text"))
                for img in page.get_images(full=True):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.alpha:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    images.append(pix.tobytes("png"))
    return texts, images

def _extract_company_name(texts: List[str]) -> str:
    text = "\n".join(texts[:3])
    prompt = f"You're reading a company's annual report. Extract the full official company name.\n\n{text[:2000]}"
    return text_llm.invoke(prompt).content.strip()

def _text_chunks(pages: List[str], size=10000, overlap=1000) -> List[str]:
    big_text = "\n".join(pages)
    return [big_text[i:i + size] for i in range(0, len(big_text), size - overlap)]

def build_faiss_index(text_chunks: List[str], img_bytes: List[bytes]):
    if os.path.exists("faiss_mm"):
        shutil.rmtree("faiss_mm")
    embedder = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=GOOGLE_API_KEY)
    docs = [(chunk, {"type": "text", "id": f"text-{i}"}) for i, chunk in enumerate(text_chunks)]
    for i, b in enumerate(img_bytes):
        b64 = base64.b64encode(b).decode()
        docs.append((b64, {"type": "image", "id": f"img-{i}"}))
    texts, metadatas = zip(*docs)
    FAISS.from_texts(texts, embedder, metadatas=list(metadatas)).save_local("faiss_mm")

def _pdf_buf_from_text(text: str, title: str) -> io.BytesIO:
    buf = io.BytesIO()
    pdf = rl_canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    pdf.setTitle(title)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, h - 50, title)
    pdf.setFont("Helvetica", 11)
    tx = pdf.beginText(40, h - 80)
    for line in text.split("\n"):
        for sub in re.findall(".{1,100}", line):
            tx.textLine(sub)
    pdf.drawText(tx)
    pdf.showPage()
    pdf.save()
    buf.seek(0)
    return buf

def summarize_performance(pages: List[str], imgs: List[bytes]) -> str:
    prompt = (
        "You are a financial analyst. Carefully review the following multimodal annual report. Provide a summary of: (a) revenue & profit trends, (b) key segments, (c) strategic initiatives, (d) major risks."
    )
    content = [prompt + "\n" + "\n".join(pages)[:18000]]
    for b in imgs[:4]:
        img_b64 = base64.b64encode(b).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
    return vision_llm.invoke([HumanMessage(content=content)]).content

def _extract_numeric(question: str, db) -> Dict[int, float]:
    chain = load_qa_chain(
        llm=text_llm,
        chain_type="stuff",
        prompt=PromptTemplate(
            template="Extract exact yearly numbers:\n{context}\n\nQuery: {question}\n\nReturn 'Year: Value' list only.",
            input_variables=["context", "question"],
        ),
    )
    docs = db.similarity_search(question, k=4)
    raw = chain({"input_documents": docs, "question": question}, return_only_outputs=True)["output_text"]
    return {int(y): float(v.replace(",", "")) for y, v in re.findall(r"(\d{4})[:\-\u2013]\s*([\d,.]+)", raw)}

def compute_ratios(db) -> pd.DataFrame:
    fields = {
        "Revenue": "Total Revenue",
        "Net Income": "Net Income or Profit",
        "Total Assets": "Total Assets",
        "Total Liabilities": "Total Liabilities",
        "Equity": "Shareholder's Equity",
        "Current Assets": "Total current Assets",
        "Current Liabilities": "Total current Liabilities",
        "Cash": "Cash and cash equivalents",
    }
    data = {k: _extract_numeric(q, db) for k, q in fields.items()}
    years = sorted({y for d in data.values() for y in d})
    rows = []
    for y in years:
        try:
            row = {
                "Year": y,
                "ROI": round(data["Net Income"].get(y, 0) / data["Total Assets"].get(y, 1), 2),
                "Debt Ratio": round(data["Total Liabilities"].get(y, 0) / data["Total Assets"].get(y, 1), 2),
                "D/E": round(data["Total Liabilities"].get(y, 0) / data["Equity"].get(y, 1), 2),
                "Current Ratio": round(data["Current Assets"].get(y, 0) / data["Current Liabilities"].get(y, 1), 2),
                "Quick Ratio": round((data["Current Assets"].get(y, 0) - data["Cash"].get(y, 0)) / data["Current Liabilities"].get(y, 1), 2),
                "Net Margin": round(data["Net Income"].get(y, 0) / data["Revenue"].get(y, 1), 2)
            }
        except:
            continue
        rows.append(row)
    return pd.DataFrame(rows).set_index("Year")

def get_market_sentiment(company_name: str) -> str:
    url = f"https://news.google.com/search?q={company_name.replace(' ', '%20')}%20financial"
    headers = {"User-Agent": "Mozilla/5.0"}
    soup = BeautifulSoup(requests.get(url, headers=headers).text, "html.parser")
    headlines = [h.text for h in soup.select("article h3")][:5]
    joined = "\n".join(headlines)
    prompt = f"Analyze the sentiment (positive/negative/neutral) of these headlines about {company_name}:\n{joined}"
    return text_llm.invoke(prompt).content

def generate_forecast(ratios_df: pd.DataFrame, company_name: str, sentiment: str) -> str:
    prompt = (
        f"You are an investment analyst. Based on this company's financial ratios and market sentiment, provide a brief forecast.\n\n"
        f"Company: {company_name}\n\n"
        f"Sentiment: {sentiment}\n\n"
        f"Ratios:\n{ratios_df.to_string()}"
    )
    return text_llm.invoke(prompt).content

# Agent Tools
tools = [
    Tool(
        name="SummarizePerformanceMM",
        func=lambda _: summarize_performance(st.session_state.pages, st.session_state.images),
        description="Summarize company performance with multimodal context."
    )
]
agent = initialize_agent(tools=tools, llm=text_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Main Interaction
if pdfs and "pages" not in st.session_state:
    with st.spinner("🔄 Processing report contents..."):
        pages, imgs = _extract_pdf_contents(pdfs)
        st.session_state.pages = pages
        st.session_state.images = imgs
        st.session_state.company_name = _extract_company_name(pages)
        build_faiss_index(_text_chunks(pages), imgs)
        st.session_state.summary = summarize_performance(pages, imgs)
        embedder = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=GOOGLE_API_KEY)
        st.session_state.db = FAISS.load_local("faiss_mm", embedder, allow_dangerous_deserialization=True)
        st.session_state.ratios = compute_ratios(st.session_state.db)
        agent.run("Run SummarizePerformanceMM")

if "summary" in st.session_state:
    st.subheader("🧠 Performance Summary")
    st.markdown(st.session_state.summary)
    st.download_button("📥 Download Summary PDF", data=_pdf_buf_from_text(st.session_state.summary, "Performance Summary"), file_name="Performance_Summary.pdf")

if "ratios" in st.session_state:
    st.subheader("📊 Financial Ratios")
    st.dataframe(st.session_state.ratios)
    st.download_button("📥 Download Ratios PDF", data=_pdf_buf_from_text(st.session_state.ratios.to_string(), "Financial Ratios"), file_name="Financial_Ratios.pdf")
    interp_prompt = (
        "You are a financial analyst. Given the ratio table below, summarize the financial health in 150 words.\n" + st.session_state.ratios.to_string()
    )
    st.markdown(text_llm.invoke(interp_prompt).content)

    st.subheader("🔮 Market Outlook")
    cname = st.text_input("Enter/Confirm Company Name:", value=st.session_state.company_name)
    if cname:
        sentiment = get_market_sentiment(cname)
        forecast = generate_forecast(st.session_state.ratios, cname, sentiment)
        st.markdown(f"**📈 Forecast for {cname}:**\n\n{forecast}")

if "db" in st.session_state:
    st.markdown("---")
    st.subheader("💬 Ask Anything About the Report")
    q = st.text_input("Ask a question:")
    if q:
        docs = st.session_state.db.similarity_search(q, k=4)
        content = ["Answer the question using the following:"]
        for d in docs:
            if d.metadata.get("type") == "text":
                content.append(d.page_content)
            else:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{d.page_content}"}})
        content.append(f"Question: {q}")
        msg = [HumanMessage(content=content)]
        st.markdown(vision_llm.invoke(msg).content)