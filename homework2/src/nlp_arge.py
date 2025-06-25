# NLP Haber Başlıkları Analizi - Kapsamlı Proje
# Kaggle News Category Dataset ile Duygu Analizi ve Topic Modeling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import string
import warnings
import os
warnings.filterwarnings('ignore')

# NLP kütüphaneleri
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

# Vektörleştirme
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Duygu analizi
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Görselleştirme
import wordcloud
from wordcloud import WordCloud

# Klasör yollarını tanımla
DATA_DIR = '../data'
RESULTS_DIR = '../results'
FIGURES_DIR = '../results/figures'
DATA_RESULTS_DIR = '../results/data'

# Klasörlerin varlığını kontrol et ve oluştur
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_RESULTS_DIR, exist_ok=True)

# Gerekli NLTK verilerini indir
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

print("="*80)
print("KAGGLE NEWS CATEGORY DATASET - KAPSAMLI NLP ANALİZİ")
print("="*80)

# Veri yükleme
try:
    df = pd.read_json(os.path.join(DATA_DIR, 'News_Category_Dataset.json'), lines=True)
    df = df[['headline', 'category']]
    print("Veri başarıyla yüklendi!")
except FileNotFoundError:
    print("Hata: News_Category_Dataset.json dosyası bulunamadı!")
    print("Lütfen dosyanın doğru konumda olduğundan emin olun.")
    exit()

print(f"\n=== VERİ SETİ GENEL BİLGİLERİ ===")
print(f"Toplam haber sayısı: {len(df):,}")
print(f"Kategori sayısı: {df['category'].nunique()}")
print(f"Eksik değer sayısı: {df.isnull().sum().sum()}")

# Kategori dağılımı
category_counts = df['category'].value_counts()

# Başlık uzunluğu istatistikleri
df['headline_length'] = df['headline'].str.len()
df['word_count'] = df['headline'].str.split().str.len()

print(f"\n=== BAŞLIK İSTATİSTİKLERİ ===")
print(f"Ortalama başlık uzunluğu: {df['headline_length'].mean():.1f} karakter")
print(f"Ortalama kelime sayısı: {df['word_count'].mean():.1f} kelime")

# Stopwords yükle
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Lemmatizer initialize
lemmatizer = WordNetLemmatizer()

print("\n" + "="*80)
print("=== ADIM 1: NLP ÖN İŞLEME ADIMLARI ===")
print("="*80)

def preprocess_text(text, use_pos_filter=True):
    """Kapsamlı ön işleme pipeline'ı"""
    if pd.isna(text):
        return ""

    # 1. String'e çevir ve küçük harfe çevir
    text = str(text).lower()

    # 2. Tokenization
    tokens = word_tokenize(text)

    # 3. Noktalama ve özel karakterleri kaldır
    tokens = [token for token in tokens if token.isalpha()]

    # 4. Stopword removal
    tokens = [token for token in tokens if token not in stop_words]

    # 5. Çok kısa kelimeleri kaldır (2 karakterden az)
    tokens = [token for token in tokens if len(token) > 2]

    # 6. Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # 7. POS filtering (isteğe bağlı)
    if use_pos_filter and tokens:
        try:
            pos_tags = pos_tag(tokens)
            # Sadece isim, sıfat, fiil türündeki kelimeleri tut
            allowed_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS',
                          'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
            tokens = [word for word, pos in pos_tags if pos in allowed_pos]
        except:
            pass  # POS tagging hatası durumunda devam et

    return ' '.join(tokens)

print("Ön işleme uygulanıyor...")
df['processed_headline'] = df['headline'].apply(lambda x: preprocess_text(x, use_pos_filter=True))

# Boş başlıkları filtrele
df = df[df['processed_headline'].str.len() > 0].reset_index(drop=True)
print(f"✓ Ön işleme tamamlandı. Kalan başlık sayısı: {len(df):,}")

# Ön işleme sonrası istatistikler
df['processed_word_count'] = df['processed_headline'].str.split().str.len()

print("\n" + "="*80)
print("=== ADIM 2: VEKTÖRLEŞTİRME YÖNTEMLERİ ===")
print("="*80)

processed_texts = df['processed_headline'].tolist()

# CountVectorizer
count_vectorizer = CountVectorizer(
    max_features=1000,
    min_df=5,
    max_df=0.7,
    ngram_range=(1, 2)
)

count_matrix = count_vectorizer.fit_transform(processed_texts)
count_feature_names = count_vectorizer.get_feature_names_out()

print(f"Count Matrix Boyutu: {count_matrix.shape}")
print(f"Kelime Hazinesi Boyutu: {len(count_feature_names)}")

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=5,
    max_df=0.7,
    ngram_range=(1, 2),
    sublinear_tf=True
)

tfidf_matrix = tfidf_vectorizer.fit_transform(processed_texts)
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

print(f"TF-IDF Matrix Boyutu: {tfidf_matrix.shape}")

# En önemli kelimeleri bul
def get_top_words(vectorizer, matrix, n_words=15):
    """En yüksek skorlu kelimeleri bul"""
    sum_words = np.array(matrix.sum(axis=0)).flatten()
    words_freq = [(word, sum_words[idx]) for word, idx in vectorizer.vocabulary_.items()]
    return sorted(words_freq, key=lambda x: x[1], reverse=True)[:n_words]

count_top_words = get_top_words(count_vectorizer, count_matrix)
tfidf_top_words = get_top_words(tfidf_vectorizer, tfidf_matrix)

print("\n" + "="*80)
print("=== ADIM 3: DUYGU ANALİZİ (SENTIMENT ANALYSIS) ===")
print("="*80)

def analyze_sentiment_textblob(text):
    """TextBlob ile duygu analizi"""
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.1:
        sentiment = 'Pozitif'
    elif polarity < -0.1:
        sentiment = 'Negatif'
    else:
        sentiment = 'Nötr'

    return polarity, subjectivity, sentiment

# Büyük veri seti için batch processing
print("TextBlob duygu analizi uygulanıyor...")
batch_size = 1000
sentiments_tb = []

for i in range(0, len(df), batch_size):
    batch = df['headline'].iloc[i:i+batch_size]
    batch_results = batch.apply(analyze_sentiment_textblob)
    sentiments_tb.extend(batch_results.tolist())

df['tb_polarity'] = [s[0] for s in sentiments_tb]
df['tb_subjectivity'] = [s[1] for s in sentiments_tb]
df['tb_sentiment'] = [s[2] for s in sentiments_tb]

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text):
    """VADER ile duygu analizi"""
    scores = analyzer.polarity_scores(str(text))

    if scores['compound'] >= 0.05:
        sentiment = 'Pozitif'
    elif scores['compound'] <= -0.05:
        sentiment = 'Negatif'
    else:
        sentiment = 'Nötr'

    return scores['compound'], sentiment

print("VADER duygu analizi uygulanıyor...")
sentiments_vader = []

for i in range(0, len(df), batch_size):
    batch = df['headline'].iloc[i:i+batch_size]
    batch_results = batch.apply(analyze_sentiment_vader)
    sentiments_vader.extend(batch_results.tolist())

df['vader_compound'] = [s[0] for s in sentiments_vader]
df['vader_sentiment'] = [s[1] for s in sentiments_vader]

# Duygu analizi sonuçları
tb_counts = df['tb_sentiment'].value_counts()
tb_percentages = df['tb_sentiment'].value_counts(normalize=True) * 100
vader_counts = df['vader_sentiment'].value_counts()
vader_percentages = df['vader_sentiment'].value_counts(normalize=True) * 100

# Uyum analizi
agreement = (df['tb_sentiment'] == df['vader_sentiment']).mean()

# Kategori bazında duygu analizi
category_sentiment = df.groupby('category').agg({
    'tb_polarity': 'mean',
    'vader_compound': 'mean'
}).round(3).sort_values('tb_polarity', ascending=False)

print("\n" + "="*80)
print("=== ADIM 4: TOPIC MODELING (KONU MODELLEME) ===")
print("="*80)

# LDA için optimal parametreler
n_topics = min(10, df['category'].nunique())
n_top_words = 8

# LDA için CountVectorizer
lda_vectorizer = CountVectorizer(
    max_features=500,
    min_df=10,
    max_df=0.5,
    ngram_range=(1, 1),
    stop_words='english'
)

lda_matrix = lda_vectorizer.fit_transform(processed_texts)
lda_feature_names = lda_vectorizer.get_feature_names_out()

# LDA modeli
lda_model = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    max_iter=20,
    learning_method='batch'
)

lda_model.fit(lda_matrix)

def display_topics(model, feature_names, n_top_words=8):
    """Konu-kelime dağılımını göster"""
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        top_weights = [topic[i] for i in top_words_idx]

        print(f"\nKonu {topic_idx + 1}:")
        words_str = ", ".join([f"{word}({weight:.2f})" for word, weight in zip(top_words, top_weights)])
        print(f"  {words_str}")

# NMF için TF-IDF
nmf_vectorizer = TfidfVectorizer(
    max_features=500,
    min_df=10,
    max_df=0.5,
    ngram_range=(1, 1),
    stop_words='english'
)

nmf_matrix = nmf_vectorizer.fit_transform(processed_texts)
nmf_feature_names = nmf_vectorizer.get_feature_names_out()

# NMF modeli
nmf_model = NMF(
    n_components=n_topics,
    random_state=42,
    max_iter=200
)

nmf_model.fit(nmf_matrix)

# Her döküman için baskın konuyu bul
lda_doc_topics = lda_model.transform(lda_matrix)
df['dominant_topic_lda'] = lda_doc_topics.argmax(axis=1)
df['topic_probability_lda'] = lda_doc_topics.max(axis=1)

nmf_doc_topics = nmf_model.transform(nmf_matrix)
df['dominant_topic_nmf'] = nmf_doc_topics.argmax(axis=1)
df['topic_probability_nmf'] = nmf_doc_topics.max(axis=1)

print("\n" + "="*80)
print("=== ADIM 5: GÖRSELLEŞTİRME VE ANALİZ ===")
print("="*80)

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# 1. Kategori dağılımı
plt.figure(figsize=(15, 8))
top_categories = category_counts.head(15)
sns.barplot(x=top_categories.values, y=top_categories.index, palette='viridis')
plt.title('En Popüler 15 Haber Kategorisi', fontsize=16, fontweight='bold')
plt.xlabel('Haber Sayısı', fontsize=12)
plt.ylabel('Kategori', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'kategori_dagilimi.png'), dpi=300, bbox_inches='tight')
plt.show()

# 2. Başlık uzunluğu dağılımı
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Karakter sayısı dağılımı
ax1.hist(df['headline_length'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax1.set_title('Başlık Uzunluğu Dağılımı (Karakter)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Karakter Sayısı')
ax1.set_ylabel('Frekans')
ax1.axvline(df['headline_length'].mean(), color='red', linestyle='--', 
            label=f'Ortalama: {df["headline_length"].mean():.1f}')
ax1.legend()

# Kelime sayısı dağılımı
ax2.hist(df['word_count'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
ax2.set_title('Kelime Sayısı Dağılımı', fontsize=14, fontweight='bold')
ax2.set_xlabel('Kelime Sayısı')
ax2.set_ylabel('Frekans')
ax2.axvline(df['word_count'].mean(), color='red', linestyle='--',
            label=f'Ortalama: {df["word_count"].mean():.1f}')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'baslik_uzunluk_dagilimi.png'), dpi=300, bbox_inches='tight')
plt.show()

# 3. Duygu analizi görselleştirmesi
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# TextBlob duygu dağılımı
tb_sentiment_counts = df['tb_sentiment'].value_counts()
colors = ['#2E8B57', '#FF6B6B', '#4682B4']
ax1.pie(tb_sentiment_counts.values, labels=tb_sentiment_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
ax1.set_title('TextBlob Duygu Dağılımı', fontsize=14, fontweight='bold')

# VADER duygu dağılımı
vader_sentiment_counts = df['vader_sentiment'].value_counts()
ax2.pie(vader_sentiment_counts.values, labels=vader_sentiment_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90)
ax2.set_title('VADER Duygu Dağılımı', fontsize=14, fontweight='bold')

# TextBlob polarity dağılımı
ax3.hist(df['tb_polarity'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax3.set_title('TextBlob Polarity Dağılımı', fontsize=14, fontweight='bold')
ax3.set_xlabel('Polarity Skoru')
ax3.set_ylabel('Frekans')
ax3.axvline(0, color='red', linestyle='--', alpha=0.7)

# VADER compound dağılımı
ax4.hist(df['vader_compound'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
ax4.set_title('VADER Compound Dağılımı', fontsize=14, fontweight='bold')
ax4.set_xlabel('Compound Skoru')
ax4.set_ylabel('Frekans')
ax4.axvline(0, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'duygu_analizi_sonuclari.png'), dpi=300, bbox_inches='tight')
plt.show()

# 4. Kategori bazında duygu analizi
plt.figure(figsize=(15, 10))
top_categories_sentiment = category_sentiment.head(15)
colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in top_categories_sentiment['tb_polarity']]

bars = plt.barh(range(len(top_categories_sentiment)), top_categories_sentiment['tb_polarity'], 
                color=colors, alpha=0.7)
plt.yticks(range(len(top_categories_sentiment)), top_categories_sentiment.index)
plt.xlabel('Ortalama Polarity Skoru', fontsize=12)
plt.title('Kategori Bazında Duygu Analizi (TextBlob)', fontsize=16, fontweight='bold')
plt.axvline(0, color='black', linestyle='-', alpha=0.3)
plt.grid(axis='x', alpha=0.3)

# Değerleri çubukların üzerine yazdır
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.01 if width >= 0 else width - 0.03, 
             bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'kategori_duygu_analizi.png'), dpi=300, bbox_inches='tight')
plt.show()

# 5. Word Cloud oluşturma
print("Word Cloud oluşturuluyor...")

# Tüm metinleri birleştir
all_text = ' '.join(processed_texts)

# Word Cloud
wordcloud = WordCloud(
    width=1200, 
    height=800,
    background_color='white',
    max_words=200,
    colormap='viridis',
    contour_width=3,
    contour_color='steelblue'
).generate(all_text)

plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Haber Başlıkları Word Cloud', fontsize=20, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'wordcloud.png'), dpi=300, bbox_inches='tight')
plt.show()

# 6. Topic Modeling görselleştirmesi
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# LDA konu dağılımı
lda_topic_counts = df['dominant_topic_lda'].value_counts().sort_index()
ax1.bar(range(len(lda_topic_counts)), lda_topic_counts.values, 
        color='lightblue', alpha=0.7, edgecolor='black')
ax1.set_title('LDA Konu Dağılımı', fontsize=14, fontweight='bold')
ax1.set_xlabel('Konu ID')
ax1.set_ylabel('Döküman Sayısı')
ax1.set_xticks(range(len(lda_topic_counts)))
ax1.set_xticklabels([f'Konu {i+1}' for i in range(len(lda_topic_counts))])

# NMF konu dağılımı
nmf_topic_counts = df['dominant_topic_nmf'].value_counts().sort_index()
ax2.bar(range(len(nmf_topic_counts)), nmf_topic_counts.values,
        color='lightcoral', alpha=0.7, edgecolor='black')
ax2.set_title('NMF Konu Dağılımı', fontsize=14, fontweight='bold')
ax2.set_xlabel('Konu ID')
ax2.set_ylabel('Döküman Sayısı')
ax2.set_xticks(range(len(nmf_topic_counts)))
ax2.set_xticklabels([f'Konu {i+1}' for i in range(len(nmf_topic_counts))])

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'topic_modeling_dagilimi.png'), dpi=300, bbox_inches='tight')
plt.show()

# 7. Vektörleştirme karşılaştırması
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Count Vectorizer top words
count_words, count_freqs = zip(*count_top_words[:10])
ax1.barh(range(len(count_words)), count_freqs, color='skyblue', alpha=0.7)
ax1.set_yticks(range(len(count_words)))
ax1.set_yticklabels(count_words)
ax1.set_title('Count Vectorizer - En Sık Kullanılan Kelimeler', fontsize=14, fontweight='bold')
ax1.set_xlabel('Frekans')

# TF-IDF top words
tfidf_words, tfidf_scores = zip(*tfidf_top_words[:10])
ax2.barh(range(len(tfidf_words)), tfidf_scores, color='lightgreen', alpha=0.7)
ax2.set_yticks(range(len(tfidf_words)))
ax2.set_yticklabels(tfidf_words)
ax2.set_title('TF-IDF - En Yüksek Skorlu Kelimeler', fontsize=14, fontweight='bold')
ax2.set_xlabel('TF-IDF Skoru')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'vektorleştirme_karsilastirma.png'), dpi=300, bbox_inches='tight')
plt.show()

print("🎯 Analiz tamamlandı!")

# Sonuçları CSV olarak kaydet
df.to_csv(os.path.join(DATA_RESULTS_DIR, 'nlp_analiz_sonuclari.csv'), index=False, encoding='utf-8')
print("✓ Sonuçlar 'results/data/nlp_analiz_sonuclari.csv' dosyasına kaydedildi.")
print("✓ Görselleştirmeler 'results/figures/' klasörüne PNG formatında kaydedildi.")