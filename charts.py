import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

# ── Color palette ──────────────────────────────────────────────
PALETTE = ['#FF6B35', '#004E89', '#1A936F', '#F18F01', '#C84B31',
           '#3A86FF', '#8338EC', '#FB5607', '#06D6A0', '#FFBE0B']
BG = '#0F1923'
CARD_BG = '#1A2634'
TEXT = '#E8EDF2'
ACCENT = '#FF6B35'

def _style(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT, labelsize=10)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2E3F50')

# 1. PIE CHART — Score Distribution
def pie_chart(df):
    counts = df['score'].value_counts().sort_index()
    labels = [f'★ {int(s)}' for s in counts.index]
    colors = ['#C84B31','#F18F01','#FFBE0B','#1A936F','#06D6A0']
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG)
    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=140,
        textprops={'color': TEXT, 'fontsize': 11},
        wedgeprops={'edgecolor': BG, 'linewidth': 2}
    )
    for t in autotexts:
        t.set_color(BG)
        t.set_fontweight('bold')
    ax.set_title('⭐ Score Distribution', color=TEXT, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig

# 2. HISTOGRAM — Review Length
def histogram(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    _style(fig, ax)
    ax.hist(df['review_length'].clip(upper=3000), bins=40, color=ACCENT, edgecolor=BG, alpha=0.85)
    ax.set_title('📝 Review Length Distribution', color=TEXT, fontsize=13, fontweight='bold')
    ax.set_xlabel('Review Length (characters)', color=TEXT)
    ax.set_ylabel('Number of Reviews', color=TEXT)
    plt.tight_layout()
    return fig

# 3. LINE CHART — Reviews Over Time
def line_chart(df):
    monthly = df.groupby('year_month').size().reset_index(name='count')
    monthly = monthly.sort_values('year_month').tail(60)
    fig, ax = plt.subplots(figsize=(8, 4))
    _style(fig, ax)
    ax.plot(monthly['year_month'], monthly['count'], color=ACCENT, linewidth=2.5, marker='o', markersize=3)
    ax.fill_between(range(len(monthly)), monthly['count'], alpha=0.15, color=ACCENT)
    ax.set_xticks(range(0, len(monthly), max(1, len(monthly)//8)))
    ax.set_xticklabels(monthly['year_month'].iloc[::max(1, len(monthly)//8)], rotation=45, ha='right', fontsize=9)
    ax.set_title('📅 Reviews Over Time', color=TEXT, fontsize=13, fontweight='bold')
    ax.set_xlabel('Month', color=TEXT)
    ax.set_ylabel('Number of Reviews', color=TEXT)
    plt.tight_layout()
    return fig

# 4. BAR CHART — Avg Score by Year
def bar_chart(df):
    yearly = df.groupby('year')['score'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    _style(fig, ax)
    bars = ax.bar(yearly['year'].astype(str), yearly['score'],
                  color=PALETTE[:len(yearly)], edgecolor=BG, linewidth=1.5)
    for bar, val in zip(bars, yearly['score']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', color=TEXT, fontsize=9, fontweight='bold')
    ax.set_ylim(0, 6)
    ax.set_title('📊 Average Score by Year', color=TEXT, fontsize=13, fontweight='bold')
    ax.set_xlabel('Year', color=TEXT)
    ax.set_ylabel('Average Score', color=TEXT)
    plt.tight_layout()
    return fig

# 5. SCATTER PLOT — Word Count vs Score
def scatter_plot(df):
    sample = df.sample(min(2000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 5))
    _style(fig, ax)
    scatter = ax.scatter(sample['word_count'], sample['score'],
                         c=sample['score'], cmap='RdYlGn',
                         alpha=0.4, s=15, edgecolors='none')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.tick_params(colors=TEXT)
    cbar.set_label('Score', color=TEXT)
    ax.set_title('🔍 Word Count vs Score', color=TEXT, fontsize=13, fontweight='bold')
    ax.set_xlabel('Word Count', color=TEXT)
    ax.set_ylabel('Score (1–5)', color=TEXT)
    ax.set_xlim(0, 1000)
    plt.tight_layout()
    return fig

# 6. BOX PLOT — Score Distribution per Sentiment
def box_plot(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    _style(fig, ax)
    order = ['Positive', 'Neutral', 'Negative']
    colors_map = {'Positive': '#06D6A0', 'Neutral': '#FFBE0B', 'Negative': '#C84B31'}
    for i, sent in enumerate(order):
        data = df[df['sentiment'] == sent]['score'].dropna()
        if len(data):
            bp = ax.boxplot(data, positions=[i], widths=0.5, patch_artist=True,
                           boxprops=dict(facecolor=colors_map[sent], color=TEXT, alpha=0.8),
                           medianprops=dict(color=BG, linewidth=2),
                           whiskerprops=dict(color=TEXT),
                           capprops=dict(color=TEXT),
                           flierprops=dict(marker='o', color=colors_map[sent], alpha=0.3, markersize=3))
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(order, color=TEXT)
    ax.set_title('📦 Score Box Plot by Sentiment', color=TEXT, fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', color=TEXT)
    plt.tight_layout()
    return fig

# 7. HEATMAP — Correlation Matrix
def heatmap(df):
    cols = ['score', 'review_length', 'word_count', 'helpfulness_ratio', 'helpful_votes', 'total_votes']
    available = [c for c in cols if c in df.columns]
    corr = df[available].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    _style(fig, ax)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                ax=ax, linewidths=0.5, linecolor='#2E3F50',
                cbar_kws={'label': 'Correlation'},
                annot_kws={'size': 10, 'color': 'white'})
    ax.set_title('🔥 Feature Correlation Heatmap', color=TEXT, fontsize=13, fontweight='bold')
    ax.tick_params(colors=TEXT)
    plt.tight_layout()
    return fig

# 8. AREA CHART — Monthly Review Volume
def area_chart(df):
    monthly = df.groupby(['year_month', 'sentiment']).size().reset_index(name='count')
    pivot = monthly.pivot(index='year_month', columns='sentiment', values='count').fillna(0).sort_index().tail(48)
    fig, ax = plt.subplots(figsize=(8, 4))
    _style(fig, ax)
    colors_area = {'Positive': '#06D6A0', 'Neutral': '#FFBE0B', 'Negative': '#C84B31'}
    x = range(len(pivot))
    for sent in ['Negative', 'Neutral', 'Positive']:
        if sent in pivot.columns:
            ax.fill_between(x, pivot[sent], alpha=0.6,
                           label=sent, color=colors_area.get(sent, ACCENT))
    ax.set_xticks(range(0, len(pivot), max(1, len(pivot)//8)))
    ax.set_xticklabels(pivot.index[::max(1, len(pivot)//8)], rotation=45, ha='right', fontsize=9)
    ax.set_title('📈 Sentiment Trends Over Time', color=TEXT, fontsize=13, fontweight='bold')
    ax.set_xlabel('Month', color=TEXT)
    ax.set_ylabel('Review Count', color=TEXT)
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT, fontsize=10)
    plt.tight_layout()
    return fig

# 9. COUNT PLOT — Reviews per Sentiment
def count_plot(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    _style(fig, ax)
    order = ['Positive', 'Neutral', 'Negative']
    counts = df['sentiment'].value_counts().reindex(order, fill_value=0)
    colors_c = ['#06D6A0', '#FFBE0B', '#C84B31']
    bars = ax.bar(order, counts.values, color=colors_c, edgecolor=BG, linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,}', ha='center', va='bottom', color=TEXT, fontsize=11, fontweight='bold')
    ax.set_title('😊 Review Count by Sentiment', color=TEXT, fontsize=13, fontweight='bold')
    ax.set_xlabel('Sentiment', color=TEXT)
    ax.set_ylabel('Count', color=TEXT)
    plt.tight_layout()
    return fig

# 10. VIOLIN PLOT — Score Distribution by Year
def violin_plot(df):
    top_years = df['year'].value_counts().nlargest(6).index.tolist()
    subset = df[df['year'].isin(sorted(top_years))]
    fig, ax = plt.subplots(figsize=(8, 5))
    _style(fig, ax)
    years_sorted = sorted(subset['year'].unique())
    data_by_year = [subset[subset['year'] == y]['score'].dropna().values for y in years_sorted]
    parts = ax.violinplot(data_by_year, positions=range(len(years_sorted)),
                         showmedians=True, showextrema=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(PALETTE[i % len(PALETTE)])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color(TEXT)
    parts['cmaxes'].set_color(TEXT)
    parts['cmins'].set_color(TEXT)
    parts['cbars'].set_color(TEXT)
    ax.set_xticks(range(len(years_sorted)))
    ax.set_xticklabels([str(y) for y in years_sorted])
    ax.set_title('🎻 Score Violin Plot by Year', color=TEXT, fontsize=13, fontweight='bold')
    ax.set_xlabel('Year', color=TEXT)
    ax.set_ylabel('Score', color=TEXT)
    plt.tight_layout()
    return fig
