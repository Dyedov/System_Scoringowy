# System Scoringowy Klientów – Segmentacja i Klasyfikacja

**Opis:**  
Ten projekt to kompleksowy system analizy klientów oparty na danych marketingowych i zakupowych. Celem projektu było stworzenie narzędzia, które pomaga firmie lepiej zrozumieć swoich klientów. Podzieliłem klientów na grupy (segmenty) według ich zachowań zakupowych, a następnie zbudowałem model uczenia maszynowego, który potrafi przewidzieć, do którego segmentu trafi nowy klient na podstawie jego cech demograficznych i zakupowych. Całość opiera się na danych z bazy Oracle, a wyniki prezentuję w czytelnej formie w Power BI i Streamlit.

---

## Zakres projektu 

1. **Przygotowanie bazy danych i wstępne przetwarzanie**
   - Projekt rozpocząłem od pobrania syntetycznych danych o klientach i załadowania ich do lokalnej bazy Oracle.
   - Utworzyłem użytkownika `scoring_user`, przydzieliłem mu odpowiednie uprawnienia i zaimportowałem dane do tabeli `customer_data`.
   - Następnie stworzyłem pięć widoków SQL, które oddzielały różne aspekty danych: zakupy, dane demograficzne, kampanie marketingowe, kanały zakupów oraz recency.
   - Dopiero po tej konfiguracji pobrałem dane z tych widoków do Pythona (Pandas), gdzie rozpocząłem dalszą analizę.
   - Szczegóły konfiguracji użytkownika Oracle i tworzenia widoków znajdują się w pliku setup_oracle.sql

2. **Analiza eksploracyjna (EDA)**
   - Na początek sprawdziłem zależności między kategoriami zakupów – stworzyłem heatmapę korelacji przy użyciu `seaborn.heatmap`.
   - Obliczyłem średnie wydatki w zależności od wieku, dochodu, liczby dzieci oraz kanału zakupów (online/offline).
   - Na tej podstawie przypisałem klientom etykiety: VIP, Pasywny lub Promocyjny.

3. **Segmentacja klientów (klasteryzacja)**
   - Przeprowadziłem segmentację klientów na podstawie wszystkich kategorii wydatków.
   - Zastosowałem algorytm KMeans, a liczbę segmentów dobrałem na podstawie wykresu łokcia (Elbow Method) stworzonego w `matplotlib`.
   - Każdy segment opisałem na podstawie dominujących cech zakupowych i demograficznych.

4. **Przygotowanie danych do klasyfikacji**
   - Do klasyfikacji przygotowałem dane wejściowe (`X` – cechy demograficzne: wiek, dochód, liczba dzieci, kanał zakupów) oraz etykiety (`y`) – czyli przypisane wcześniej segmenty.
   - Dane podzieliłem na zbiór treningowy i testowy (80/20).

5. **Porównanie modeli klasyfikacyjnych**
   - Przetestowałem trzy modele klasyfikacji – każdy osobno, na tych samych danych:
     - **RandomForestClassifier**
     - **LogisticRegression**
     - **XGBoostClassifier**
   - Każdy z modeli oceniłem przy użyciu:
     - Accuracy
     - Confusion matrix – w dwóch wersjach: `ConfusionMatrixDisplay` z `sklearn` oraz `seaborn.heatmap`
   - Na podstawie wyników porównałem skuteczność modeli i jakość ich predykcji.

6. **Budowa dashboardów na podstawie segmentacji**
   - Po zakończeniu procesu segmentacji i klasyfikacji wróciłem do oryginalnych, niestandaryzowanych danych demograficznych i zakupowych.
   - Do tych danych dołączyłem kolumnę `SEGMENT`, wyznaczoną wcześniej na podstawie KMeans.
   - Z tak przygotowanego zestawu stworzyłem osobny plik `.csv`, który posłużył jako źródło do wizualizacji wyników.
   - Na tej podstawie zbudowałem dwa dashboardy:
     - **Power BI** – do analizy rozkładu segmentów oraz średnich wartości cech (wydatki, dochód, dzieci, recency, liczba klientów).
     - **Streamlit** – interaktywny dashboard uruchamiany z poziomu Pythona, z dynamicznymi wykresami słupkowymi dla każdego segmentu.

---

## Wizualizacje analityczne i ewaluacyjne

- Heatmapa korelacji zakupów (`seaborn.heatmap`)
- Wykres łokcia (Elbow Method) – wybór liczby klastrów (`matplotlib`)
- Confusion matrix – w dwóch wersjach: `sklearn.metrics.ConfusionMatrixDisplay` i `seaborn.heatmap`
- Wykresy słupkowe i porównania średnich wartości cech demograficznych (`matplotlib`, `seaborn`)
- Interaktywne dashboardy – Power BI i Streamlit

---

## Użyte technologie i biblioteki

- **Języki**: Python, SQL (Oracle)
- **Baza danych**: Oracle SQL (widoki: VW_ZAKUPY, VW_KAMPANIE, VW_DEMOGRAFIA, VW_ONLINE_OFFLINE, VW_RECENCY)
- **Biblioteki Python:**
  - `pandas`, `numpy` – przetwarzanie danych
  - `matplotlib`, `seaborn` – wizualizacje (heatmapy, wykres łokcia, wykresy słupkowe)
  - `sklearn` – klasyfikacja (RandomForest, LogisticRegression), ewaluacja (accuracy, confusion matrix)
  - `xgboost` – klasyfikacja (XGBoostClassifier)
  - `scipy.cluster.vq.kmeans` – klasteryzacja (KMeans)
  - `streamlit` – dashboard
  - `oracledb` – połączenie z Oracle DB
- **Narzędzia wizualizacji**: Power BI, Streamlit

---

## 📊 Przykładowe wyniki

- Segmentacja klientów: 4 segmenty nazwane ręcznie na podstawie zachowań i profilu zakupowego.
- Klasyfikacja klientów do segmentów na podstawie cech demograficznych, z porównaniem trzech modeli ML.
- Interaktywne dashboardy przedstawiające rozkłady wydatków, dochodów, liczby dzieci, recency oraz liczebność segmentów.

---

## 🖼️ Wizualizacje

### 🔹 Heatmapa korelacji wydatków
![Heatmapa korelacji](images/heatmapa_korelacji.png)

### 🔹 Wykres łokcia (Elbow Method)
![Wykres łokcia](images/wykres_lokcia.png)

### 🔹 Wykres kołowy – udział segmentów
![Wykres kołowy](images/wykres_kolowy.png)

### 🔹 Dashboard Power BI (statyczny podgląd)
![Dashboard](images/dashboard.jpg)

### 🔹 Macierze pomyłek:
- Random Forest:
  ![Confusion RF](images/confusion_rf.png)
- Logistic Regression:
  ![Confusion LR](images/confusion_lr.png)
- XGBoost:
  ![Confusion XGB](images/confusion_xgb.png)

---

## 🚀 Uruchomienie (lokalnie)

1. Skonfiguruj połączenie z bazą Oracle.
2. Uruchom skrypt `System_Scoringowy.py` w środowisku Python.
3. Aby wyeksportować dane do dashboardu: ustaw `eksportowac = True`
4. Aby uruchomić dashboard w Streamlit:
```bash
streamlit run System_Scoringowy.py
```
*(upewnij się, że `uruchom_dashboard = True` w kodzie)*

---

## 📁 Struktura plików

```
System_Scoringowy.py         # główny skrypt analizy i dashboardu
dane_powerbi_niestandard.csv # dane do dashboardów (segmentacja + dane demograficzne)
```

---

## ✍️ Autor

**Dmytro Dyedov**  
Junior Data Scientist – Python | SQL | Machine Learning  
