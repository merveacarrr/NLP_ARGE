# NLP Haber Başlıkları Analizi Projesi

Bu proje, Kaggle News Category Dataset kullanarak haber başlıkları üzerinde kapsamlı NLP (Doğal Dil İşleme) analizi gerçekleştirir.

## Proje Yapısı

```
homework2/
├── 📂 src/                          # Kaynak kodlar
│   └── nlp_arge.py                  # Ana analiz scripti
├── 📂 data/                         # Veri dosyaları
│   └── News_Category_Dataset.json    # Kaggle haber veri seti
├── 📂 results/                      # Analiz sonuçları
│   ├── 📂 figures/                  # Görselleştirmeler
│   │   ├── kategori_dagilimi.png
│   │   ├── baslik_uzunluk_dagilimi.png
│   │   ├── duygu_analizi_sonuclari.png
│   │   ├── kategori_duygu_analizi.png
│   │   ├── topic_modeling_dagilimi.png
│   │   ├── vektorleştirme_karsilastirma.png
│   │   └── wordcloud.png
│   └── 📂 data/                     # İşlenmiş veriler
│       └── nlp_analiz_sonuclari.csv
├── 📂 docs/                         # Dokümantasyon
│   └── README.md                 # Detaylı dokümantasyon
├── requirements.txt              # Python bağımlılıkları
└── README.md                     # Bu dosya
```

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.7+
- pip (Python paket yöneticisi)

### Adım 1: Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### Adım 2: Veri Setini Hazırlayın
- `News_Category_Dataset.json` dosyasını `data/` klasörüne yerleştirin
- Dosya Kaggle'dan indirilebilir: [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)

### Adım 3: Analizi Çalıştırın
```bash
cd src
python nlp_arge.py
```

## 🔍 Analiz Özellikleri

### 1. 📝 NLP Ön İşleme
- **Tokenization**: Metni kelimelere ayırma
- **Lowercasing**: Büyük/küçük harf normalizasyonu
- **Stopword Removal**: Anlamsız kelimeleri kaldırma
- **Lemmatization**: Kelimeleri kök haline getirme
- **POS Filtering**: Kelime türü filtreleme

### 2. 🔢 Vektörleştirme
- **Count Vectorizer**: Kelime frekans sayımı
- **TF-IDF Vectorizer**: Ağırlıklı kelime frekansı
- **N-gram Analizi**: 1-gram ve 2-gram desteği

### 3. 😊 Duygu Analizi
- **TextBlob**: Genel amaçlı duygu analizi
- **VADER**: Sosyal medya metinleri için optimize
- **Kategori Bazında Analiz**: Her kategori için duygu skorları

### 4. 📚 Topic Modeling
- **LDA (Latent Dirichlet Allocation)**: Olasılıksal konu modelleme
- **NMF (Non-negative Matrix Factorization)**: Matris ayrıştırma
- **Konu Dağılımı Analizi**: Döküman-konu atamaları

### 5. 📊 Görselleştirme
- Kategori dağılım grafikleri
- Başlık uzunluğu analizleri
- Duygu analizi sonuçları
- Word Cloud görselleştirmesi
- Topic modeling dağılımları
- Vektörleştirme karşılaştırmaları

## 📈 Sonuçlar

Analiz sonuçları şu dosyalarda saklanır:

### Görselleştirmeler (`results/figures/`)
- `kategori_dagilimi.png`: En popüler haber kategorileri
- `baslik_uzunluk_dagilimi.png`: Başlık uzunluğu istatistikleri
- `duygu_analizi_sonuclari.png`: Duygu analizi sonuçları
- `kategori_duygu_analizi.png`: Kategori bazında duygu analizi
- `topic_modeling_dagilimi.png`: Konu modelleme dağılımları
- `vektorleştirme_karsilastirma.png`: Vektörleştirme yöntemleri karşılaştırması
- `wordcloud.png`: Haber başlıkları word cloud

### Veri Sonuçları (`results/data/`)
- `nlp_analiz_sonuclari.csv`: Tüm analiz sonuçları (39MB)

## 🛠️ Teknik Detaylar

### Kullanılan Kütüphaneler
- **NLP**: NLTK, TextBlob, VADER
- **Veri İşleme**: Pandas, NumPy
- **Makine Öğrenmesi**: Scikit-learn
- **Görselleştirme**: Matplotlib, Seaborn, WordCloud

### Performans Optimizasyonları
- Batch processing büyük veri setleri için
- Bellek verimli vektörleştirme
- Paralel işleme desteği
- Otomatik klasör oluşturma

## 📋 Çıktı Özeti

```
PROJE SONUÇLARI:
================
📊 Veri Seti: 200,000+ haber başlığı, 40+ kategori
🔧 Ön İşleme: Tokenization → Lowercasing → Stopword Removal → Lemmatization
📈 Vektörleştirme: Count Vectorizer (1000 özellik), TF-IDF (1000 özellik)
😊 Duygu Analizi: TextBlob ve VADER karşılaştırması
📚 Topic Modeling: LDA ve NMF ile 10 konu
```

## 🤝 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun



**Not**: Bu proje, NLP tekniklerinin pratik uygulamasını göstermek amacıyla geliştirilmiştir. Ticari kullanım için ek optimizasyonlar gerekebilir. 
