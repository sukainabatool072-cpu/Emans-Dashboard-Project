# 🍽️ Amazon Fine Foods Reviews Dashboard

**Course:** Exploratory Data Analysis  
**Instructor:** Ali Hassan Sherazi  
**Submission:** 05-June-2026

---

## 📌 Project Overview

This dashboard analyzes the **Amazon Fine Foods Reviews** dataset using Python, Pandas, Matplotlib, Seaborn, and Streamlit. It presents 10 different chart types with interactive filters to explore review patterns, sentiment trends, and score distributions.

---

## 📁 Folder Structure

```
dashboard_project/
├── data/
│   └── finefoods.txt.gz       ← DO NOT rename
├── app.py                     ← Main Streamlit app
├── charts.py                  ← All chart functions
├── filters.py                 ← Filter logic reference
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the dashboard
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## 📊 Charts Included

| # | Chart Type | Insight |
|---|-----------|---------|
| 1 | Pie Chart | Score distribution (1–5 stars) |
| 2 | Histogram | Review length distribution |
| 3 | Line Chart | Monthly review volume over time |
| 4 | Bar Chart | Average score per year |
| 5 | Scatter Plot | Word count vs Score relationship |
| 6 | Box Plot | Score spread by sentiment |
| 7 | Heatmap | Feature correlation matrix |
| 8 | Area Chart | Sentiment trends over time |
| 9 | Count Plot | Review count by sentiment |
| 10 | Violin Plot | Score distribution by year |

---

## 🎛️ Filters

- **Score Range** — Slider to filter by star rating
- **Year Range** — Slider for time period
- **Sentiment** — Dropdown (All / Positive / Neutral / Negative)
- **Specific Scores** — Multi-select star ratings
- **Search** — Keyword search in review text
- **Reset** — One-click reset all filters

---

## 💡 Key Insights

1. ~70%+ of reviews are Positive (4–5 stars)
2. Review volume increased significantly from 2008–2012
3. Longer reviews tend to be more helpful
4. Score has weak correlation with review length
5. Most negative reviews are short and use fewer words
