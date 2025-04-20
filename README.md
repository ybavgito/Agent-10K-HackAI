# 🤖 Agent 10K – Multimodal AI Annual Report Analyst

Agent 10K is a multimodal AI-powered Streamlit app that analyzes corporate annual reports (10-Ks) using both text and image data. It leverages Google Gemini models, LangChain agents, and financial logic to extract insights, compute ratios, forecast trends, and answer investor-grade questions.

---

## 🚀 Features

- 📁 **Upload Multiple Annual Reports** (PDF)
- 🧠 **Multimodal Analysis**: Combines text + charts/images using Gemini Pro Vision
- 📝 **Autonomous Agent**: Summarizes financial performance via LangChain agent
- 📊 **Financial Ratio Computation**: ROI, D/E, Quick Ratio, Net Margin, and more
- 🔮 **Market Sentiment & Forecast**: Scrapes news headlines and gives AI-driven outlook
- 💬 **Ask Anything**: Semantic search and QA across reports with visual support
- 📄 **PDF Report Downloads**: Export AI-generated summaries and financials

---

## 🧱 Tech Stack

- **Python**
- **Streamlit** – interactive UI
- **LangChain** – agent orchestration & RAG
- **Google Gemini Pro / Vision API**
- **FAISS** – vector storage for text & images
- **PyMuPDF / PIL / ReportLab** – PDF parsing, image extraction, and generation
- **BeautifulSoup** – headline scraping for sentiment analysis
