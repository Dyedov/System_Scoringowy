import oracledb                    # Biblioteka do łączenia się z bazą Oracle Database.
import pandas as pd                # Pandas – do przetwarzania i analizy danych.
import seaborn as sns              # Seaborn – do wizualizacji macierzy korelacji oraz macierzy pomyłek.
import matplotlib.pyplot as plt    # Matplotlib – do wykresów.

pd.set_option('display.max_columns', None)    # Wyświetlanie wszystkich kolumn w dataframe'ach.

# --- Połączenie z bazą danych Oracle (lokalnie, użytkownik scoring_user)
connection = oracledb.connect(
    user = 'scoring_user',
    password = 'haslo123',
    dsn = 'localhost/XEPDB1'
)

# --- Przykładowe użycie kursora Oracle (można sprawdzić ile jest wierszy itp.)
# with connection.cursor() as cursor:
#     cursor.execute('SELECT COUNT(*) FROM customer_data')
#     liczba = cursor.fetchone()[0]
#     print('Liczba wierszy:', liczba)
#
#     cursor.execute('SELECT * FROM customer_data FETCH FIRST 5 ROWS ONLY')
#     for row in cursor:
#         print(row)

# --- Wylistowanie wszystkich widoków w bazie danych (np. VW_ZAKUPY itd.)
wszystkie_widoki = pd.read_sql('SELECT view_name FROM user_views', con=connection)
print('Widoki w bazie danych:')
print(wszystkie_widoki)

# --- Wczytanie głównych widoków do DataFrame (każdy widok to osobny zbiór danych)
zakupy_df = pd.read_sql('SELECT * FROM VW_ZAKUPY', con=connection)                  # Zakupy klientów
kampanie_df = pd.read_sql('SELECT * FROM VW_KAMPANIE', con=connection)              # Udział/reakcja w kampaniach marketingowych
demografia_df = pd.read_sql('SELECT * FROM VW_DEMOGRAFIA', con=connection)          # Dane demograficzne (rok urodzenia, dzieci itd.)
online_offline_df = pd.read_sql('SELECT * FROM VW_ONLINE_OFFLINE', con=connection)  # Kanały zakupów (online/offline)
recency_df = pd.read_sql('SELECT ID, RECENCY FROM CUSTOMER_DATA', con=connection)   # Dni od ostatniego zakupu

# --- Podgląd danych z widoków
# print(zakupy_df.head(11))
# print(kampanie_df.head())
# print(demografia_df.head())
# print(online_offline_df.head())

# --- Wyznaczanie korelacji wydatków + wizualizacja heatmapy (analiza eksploracyjna)
korelacja_wydatkow_heatmapa = False

if korelacja_wydatkow_heatmapa:
    wydatki_kolumny = zakupy_df.drop(columns=['ID'])
    korelacje = wydatki_kolumny.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(korelacje, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Heatmapa korelacji wydatków")
    plt.tight_layout()
    plt.show()

# --- Łączenie danych z kilku widoków (jeden rekord = jeden klient, zawiera info o wydatkach, demografii, kampaniach, kanale zakupu)
dane = zakupy_df.merge(demografia_df, on='ID')
dane = dane.merge(kampanie_df, on='ID')
dane = dane.merge(online_offline_df, on='ID')
print(dane.head())

# --- Wyliczanie wieku klientów (rok analizy = 2025)
dane['WIEK'] = 2025 - dane['ROK_URODZENIA']

# --- Obliczanie średnich wydatków na wybrane kategorie w zależności od wieku
srednie_wg_wieku = dane.groupby('WIEK')[['WYDATKI_WINO', 'WYDATKI_ZLOTO', 'WYDATKI_SLODYCZE']].mean().reset_index()
# srednie_wg_wieku = srednie_wg_wieku.sort_index(ascending=False)  # --- Opcjonalnie: sortowanie wg wieku malejąco
srednie_wg_wieku = srednie_wg_wieku.round(2)

print('\n')
print('Średnie wydatki wg wieku:')
print(srednie_wg_wieku.head())

# --- Liczba dzieci (tutaj jako maksimum z dwóch kolumn: dzieci w domu i nastolatki w domu)
dane['DZIECI'] = dane[['DZIECI_W_DOMU', 'NASTOLATKI_W_DOMU']].max(axis=1)

# --- Średnie wydatki według liczby dzieci
srednie_wg_dzieci = (
    dane.groupby('DZIECI')[['WYDATKI_WINO','WYDATKI_ZLOTO', 'WYDATKI_SLODYCZE']]
    .mean()
    .round(2)
    .reset_index()
)
print('\n')
print('Średnie wydatki wg liczby dzieci:')
print(srednie_wg_dzieci)
# print(dane['DZIECI'].value_counts())

# --- Odfiltrowanie klientów bez dochodu, binning dochodu co 10 tys.
dane = dane[dane['DOCHOD'].notna()]
dane['PRZEDZIAL_DOCHODU'] = (dane['DOCHOD'] // 10000) * 10000
dane = dane[dane['PRZEDZIAL_DOCHODU'] > 0]

# --- Średnie wydatki wg przedziału dochodu
srednie_wg_dochodu = (
    dane.groupby('PRZEDZIAL_DOCHODU')[['WYDATKI_WINO','WYDATKI_ZLOTO', 'WYDATKI_SLODYCZE']]
    .mean()
    .round(2)
    .reset_index()
)
print('\nŚrednie wydatki wg przedziału dochodu:')
print(srednie_wg_dochodu.head(10))

# --- Podgląd nagłówków kolumn z różnych widoków
print('\n')
# print(list(online_offline_df.columns))
print('Nagłówki kolumn VW_ONLINE_OFFLINE: ')
for kolumny in online_offline_df.columns:
    print(kolumny)

print('\n')
print('Nagłówki kolumn VW_ZAKUPY: ')
for kolumny in zakupy_df.columns:
    print(kolumny)

# --- Połączenie zakupów z kanałami (online/offline)
df = pd.merge(zakupy_df, online_offline_df, on = 'ID')

# --- Oznaczenie czy klient kupuje online/offline (binarnie: 1/0)
df['ONLINE'] = (df['ZAKUPY_ONLINE'] > 0).astype(int)
df['OFFLINE'] = ((df['ZAKUPY_STACJONARNE'] > 0) | (df['ZAKUPY_KATALOGOWE'] > 0)).astype(int)

# --- Średnie wydatki według kanału zakupu (online/offline)
srednie_online_offline = (
    df.groupby(['ONLINE', 'OFFLINE'])[
        ['WYDATKI_WINO', 'WYDATKI_OWOCE', 'WYDATKI_MIESO', 'WYDATKI_RYBY', 'WYDATKI_SLODYCZE', 'WYDATKI_ZLOTO']
    ]
    .mean()
    .round(2)
    .reset_index()
)
print('\nŚrednie wydatki wg kanału zakupu (ONLINE vs OFFLINE):')
print(srednie_online_offline)

# --- Podgląd nagłówków kolumn z widoku kampanii marketingowych
print('\n')
# print(list(kampanie_df.columns))
print('Nagłówki kolumn VW_KAMPANIE: ')
for kolumny in kampanie_df.columns:
    print(kolumny)

# --- Wyliczenie sumy wydatków na wszystkie kategorie (kolumna SUMA_WYDATKÓW)
df['SUMA_WYDATKÓW'] = (
    df['WYDATKI_WINO']
    + df['WYDATKI_OWOCE']
    + df['WYDATKI_MIESO']
    + df['WYDATKI_RYBY']
    + df['WYDATKI_SLODYCZE']
    + df['WYDATKI_ZLOTO']
)

# --- Oznaczenie klientów VIP (top 10% pod względem wydatków)
prog_vip = df['SUMA_WYDATKÓW'].quantile(0.90)
df['VIP'] = (df['SUMA_WYDATKÓW'] >= prog_vip).astype(int)

# --- Oznaczenie klientów pasywnych (bottom 10% pod względem wydatków)
prog_pasywny = df['SUMA_WYDATKÓW'].quantile(0.10)
df['PASYWNY'] = (df['SUMA_WYDATKÓW'] <= prog_pasywny).astype(int)

# --- Dołączenie do danych odpowiedzi klienta na kampanię marketingową (czy zareagował/promocyjny)
df = pd.merge(df, kampanie_df[['ID', 'ODPOWIEDZ_NA_KAMPANIE']], on='ID', how='left')
df['PROMOCYJNY'] = df['ODPOWIEDZ_NA_KAMPANIE'].fillna(0).astype(int)

# --- Dołączenie kolumny recency (ile dni od ostatniego zakupu)
df = pd.merge(df, recency_df, on='ID', how='left')

# --- Podgląd wybranych kolumn — przykładowi klienci z etykietami
print('\nPodsumowanie końcowe (z kolumną RECENCY):')
print(df[['ID', 'SUMA_WYDATKÓW', 'VIP', 'PASYWNY', 'PROMOCYJNY', 'RECENCY']].head(10))

# --- Podgląd recency_df (dane surowe)
print('\nrecency_df HEAD:')
print(recency_df.head())
print('\nKolumny recency_df:', recency_df.columns.tolist())

# --- Liczby klientów w każdej grupie
print('\nLiczba VIP:', df['VIP'].sum())
print('Liczba pasywnych:', df['PASYWNY'].sum())
print('Liczba promocyjnych:', df['PROMOCYJNY'].sum())

# --- Wyświetlenie nazw kolumn w każdym DataFrame (wszystkie główne widoki) - widok Listy
# print('\nWyświetlenie nazw kolumn w każdym DataFrame (wszystkie główne widoki)')
# for nazwa, df in {
# "VW_ZAKUPY": zakupy_df,
#     "VW_KAMPANIE": kampanie_df,
#     "VW_DEMOGRAFIA": demografia_df,
#     "VW_ONLINE_OFFLINE": online_offline_df
# }.items():
#     print(f'\nNagłówki kolumn {nazwa}:')
#     print(df.columns.tolist())

# --- Wyświetlenie nazw kolumn w każdym DataFrame (wszystkie główne widoki) - widok Kolumny
print('\nWyświetlenie nazw kolumn w każdym DataFrame (wszystkie główne widoki)')
dataframes = {
"VW_ZAKUPY": zakupy_df,
    "VW_KAMPANIE": kampanie_df,
    "VW_DEMOGRAFIA": demografia_df,
    "VW_ONLINE_OFFLINE": online_offline_df
}
for nazwa, df in dataframes.items():
    print(f'\nNagłówki kolumn {nazwa}:')
    for kolumna in df.columns:
        print(f'- {kolumna}')

# --- Połączenie wszystkich danych razem (zakupy, demografia, online_offline, kampanie, recency)
df = pd.merge(zakupy_df, demografia_df, on='ID', how='left')
df = pd.merge(df, online_offline_df, on='ID', how='left')
df = pd.merge(df, kampanie_df[['ID', 'ODPOWIEDZ_NA_KAMPANIE']], on='ID', how='left')
df = pd.merge(df, recency_df, on='ID', how='left')

# --- Ponowne wyliczenie sumy wydatków (na potrzeby klasteryzacji)
df['SUMA_WYDATKÓW'] = (
    df['WYDATKI_WINO']
    + df['WYDATKI_OWOCE']
    + df['WYDATKI_MIESO']
    + df['WYDATKI_RYBY']
    + df['WYDATKI_SLODYCZE']
    + df['WYDATKI_ZLOTO']
)

# --- Liczba dzieci (maksimum z dwóch kolumn)
df['LICZBA_DZIECI'] = df[['DZIECI_W_DOMU', 'NASTOLATKI_W_DOMU']].max(axis=1)

# --- Podgląd wszystkich kolumn w df (ostateczny "master" DataFrame na ten etap)
# print('\nKolumny w df:', df.columns.tolist())
print('\nKolumny w df:')
for col in df.columns:
    print(col)

from sklearn.preprocessing import StandardScaler  # StandardScaler – do standaryzacji cech przed klasteryzacją i ML (średnia=0, std=1)

# --- Lista cech używanych do klasteryzacji i ML
cechy = ['SUMA_WYDATKÓW', 'RECENCY', 'DOCHOD', 'LICZBA_DZIECI']

scaler = StandardScaler()  # Standaryzacja (średnia=0, std=1), wymagane przez większość algorytmów ML

# --- Standaryzacja wybranych cech, zachowanie kolumny ID
df_scaled = scaler.fit_transform(df[cechy])
df_scaled = pd.DataFrame(df_scaled, columns=cechy)
df_scaled['ID'] = df['ID'].values

print('\nDane po standaryzacji (pierwsze 5 wierszy):')
print(df_scaled.head())

# --- Sprawdzanie wersji bibliotek
# try:
#     import sklearn
#     import matplotlib
#     print('Biblioteka sklearn jest zainstalowana: wersja', sklearn.__version__)
#     print('Biblioteka matplotlib jest zainstalowana: wersja', matplotlib.__version__)
# except ImportError as e:
#     print('Brakuje biblioteki:', e.name)

# --- Podgląd nazw kolumn w df po merge/przetwarzaniu
print('\nKolumny w df:')
for col in df.columns:
    print(col)

from sklearn.cluster import KMeans  # KMeans – do klasteryzacji/segmentacji klientów (grupowania klientów na segmenty na podstawie cech)

# --- Przygotowanie danych do klasteryzacji (.dropna() - usuwa wiersze z brakami danych (NaN))
x = df_scaled[['SUMA_WYDATKÓW', 'RECENCY', 'DOCHOD', 'LICZBA_DZIECI']].dropna()

# --- Metoda łokcia – wybór liczby klastrów na podstawie inercji (suma kwadratów odległości do centroidów)
inercja = []
zakres_klastrow = range(1, 11)
for k in zakres_klastrow:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x)
    inercja.append(kmeans.inertia_)

# --- Wykres łokcia do oceny liczby klastrów
# plt.figure(figsize=(8, 5))
# plt.plot(zakres_klastrow, inercja, marker='o')
# plt.xlabel('Liczba klastrów (k)')
# plt.ylabel('Inercja')
# plt.title('Metoda łokcia - wybór liczby klastrów')
# plt.grid(True)
# plt.show()

# --- Finalna klasteryzacja (wybrano 4 klastry na podstawie metody łokcia)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(x)
x['KLASTR'] = kmeans.labels_

# --- Podsumowanie: średnie wartości cech w każdym klastrze
grupy = x.groupby('KLASTR').mean().reset_index()
print('\nKlasteryzacja x = df_scaled:')
print(grupy)
print()

# --- Nazwy segmentów przypisane do klastrów (ręcznie zmapowane)
segmenty = {
    0: 'Oszczędni aktywni',
    1: 'Lojalni rodzinni',
    2: 'Pasywni',
    3: 'VIP'
}
x['SEGMENT'] = x['KLASTR'].map(segmenty)

# --- Liczba klientów w każdym segmencie
print(x[['KLASTR', 'SEGMENT']].value_counts().reset_index(name='liczba_klientów'))
print()

podsumowanie_segmentow = x['SEGMENT'].value_counts().reset_index(name='LICZBA KLIENTÓW').rename(columns={'index': 'SEGMENT'})
print(podsumowanie_segmentow)

# --- Wykresy udziału segmentów (bar/pie chart)
# segmenty_liczba = x['SEGMENT'].value_counts()
# segmenty_liczba.plot(kind='bar', color='skyblue')
# plt.title('Liczba klientów w segmentach')
# plt.ylabel('Liczba klientów')
# plt.xlabel('Segment')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# --- Podgląd wyników klasteryzacji
print('\n', x.columns)
print('\n', x.head())
print('\n', x.tail())


# --- Pie chart udziałów segmentów
# plt.figure(figsize=(8, 8))
# plt.pie(
#     podsumowanie_segmentow['LICZBA KLIENTÓW'],
#     labels=podsumowanie_segmentow['SEGMENT'],
#     autopct='%1.1f%%',
#     startangle=140,
#     colors=plt.cm.Paired.colors
# )
# plt.title('Udział segmentów klientów')
# plt.axis('equal')
# plt.show()

# --- # Wyświetlenie nazw wszystkich DataFrame'ów zdefiniowanych w programie (diagnostyka/debugowanie)
# print('\nWyświetlenie wszystkich DataFrame'ów')
# for name, val in list(globals().items()):
#     if isinstance(val, pd.DataFrame):
#         print(name)

# --- Przygotowanie danych do modelowania ML (zachowanie klastrów i segmentów)
dane_model_ml = x.copy()
print('\nDANE_MODEL_ML (Nazwy Kolumn):')
for col in dane_model_ml.columns:
    print(col)
print()

x_cechy = dane_model_ml[['SUMA_WYDATKÓW', 'RECENCY', 'DOCHOD', 'LICZBA_DZIECI']]  # cechy wejściowe
y_segment = dane_model_ml['SEGMENT']                                              # etykieta do predykcji (segment)

print(y_segment.value_counts())

from sklearn.model_selection import train_test_split  # Funkcja z biblioteki scikit-learn. Podział danych na train/test do uczenia i testowania modeli ML

# --- Podział na zbiory treningowy i testowy (80/20), stratified – zachowanie proporcji segmentów
x_train, x_test, y_train, y_test = train_test_split(
    x_cechy,
    y_segment,
    test_size=0.2,
    random_state=42,
    stratify=y_segment
)

print('\n- Informacje o liczbie próbek i rozkładzie klas w zbiorach treningowym i testowym -')
print('\nLiczba próbek w zbiorach train/test:')
print('Liczba próbek treningowych:', len(x_train))
print('Liczba próbek testowych:', len(x_test))
print('\nRozkład klas w zbiorze treningowym:')
print(y_train.value_counts(normalize=True))
print('\nRozkład klas w zbiorze testowym:')
print(y_test.value_counts(normalize=True))

# --- Random Forest Classifier – model klasyfikacyjny (biblioteka: scikit-learn)
from sklearn.ensemble import RandomForestClassifier  # Random Forest – klasyfikator ML

model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# --- Ewaluacja (ocena skuteczności) modelu ML (raport, dokładność, macierz pomyłek)
from sklearn.metrics import classification_report  # classification_report - podsumowanie (wyników klasyfikacji) skuteczności modelu (precyzja, recall, F1-score) dla każdej klasy/segmentu
from sklearn.metrics import accuracy_score         # accuracy_score – sprawdza, jaki procent przewidywań był poprawny (ile razy model trafił w prawidłowy segment)
from sklearn.metrics import confusion_matrix       # confusion_matrix – macierz pomyłek (pokazuje, które klasy (segmenty) zostały pomylone (są mylone z którymi))

print('\n- Wyświetlenie wyników oceny skuteczności modelu Random Forest -')
print('\nDokładność (accuracy):', accuracy_score(y_test, y_pred))
print('\nRaport klasyfikacji:')
print(classification_report(y_test, y_pred))
print('Macierz pomyłek:')
print(confusion_matrix(y_test, y_pred))

# --- Wizualizacja macierzy pomyłek (Random Forest)
from sklearn.metrics import  ConfusionMatrixDisplay  # ConfusionMatrixDisplay - graficzne wyświetlenie macierzy pomyłek dla wyników wybranego modelu ML

macierz_pomylek_Random_Forest = False  # True - włączyć / False - wyłączyć. Graficzne wyświetlenie macierzy pomyłek ConfusionMatrixDisplay

if macierz_pomylek_Random_Forest:
    ConfusionMatrixDisplay.from_estimator(
        model,
        x_test,
        y_test,
        display_labels=model.classes_,
        cmap='Reds',
        xticks_rotation=45
    )
    plt.title('Macierz pomyłek Random Forest')
    plt.tight_layout()
    plt.show()

# --- Logistic Regression – alternatywny model klasyfikacyjny (biblioteka: scikit-learn)
from sklearn.linear_model import LogisticRegression  # Logistic Regression – klasyfikator ML

logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(x_train, y_train)
y_pred_logreg = logreg_model.predict(x_test)

print('\n- Wyświetlenie wyników oceny skuteczności modelu Logistic Regression -')
print('\nDokładność (accuracy)', accuracy_score(y_test, y_pred_logreg))
print('\nRaport klasyfikacji:')
print(classification_report(y_test, y_pred_logreg))
print('Macierz pomyłek:')
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
print(cm_logreg)

# --- Wizualizacja macierzy pomyłek (Logistic Regression)
macierz_pomylek_Logistic_Regression = False  # True - włączyć / False - wyłączyć. Graficzne wyświetlenie macierzy pomyłek Seaborn (sns.heatmap)

if macierz_pomylek_Logistic_Regression:
    segmenty = ['Lojalni rodzinni','Oszczędni aktywni','Pasywni','VIP']
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Reds',
                xticklabels=segmenty, yticklabels=segmenty)
    plt.xlabel('Predicted label')
    plt.ylabel('True Label')
    plt.title('Macierz pomyłek Logistic Regression')
    plt.tight_layout()
    plt.show()

# --- Przygotowanie etykiet do XGBoost (numeryczne)
from  sklearn.preprocessing import LabelEncoder  # LabelEncoder – kodowanie etykiet (segmentów) na liczby całkowite (wymagane przez XGBoost)

le = LabelEncoder()
y_train.enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# --- XGBoost – model klasyfikacyjny, bardzo wydajny do predykcji segmentu (biblioteka: xgboost)
from xgboost import XGBClassifier  # XGBClassifier – klasyfikator ML

# --- Trenowanie modelu XGBoost na danych treningowych
xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb_model.fit(x_train, y_train.enc)

# --- Predykcja segmentów na danych testowych (XGBoost)
y_pred_xgb = xgb_model.predict(x_test)
y_pred_xgb_labels = le.inverse_transform(y_pred_xgb)  # Zamiana przewidywanych etykiet numerycznych z powrotem na kategorie

print('\n- Wyświetlenie wyników oceny skuteczności modelu XGBoost -')
print('\nDokładność (accuracy):', accuracy_score(y_test, y_pred_xgb_labels))
print('\nRaport klasyfikacji:')
print(classification_report(y_test, y_pred_xgb_labels))
print('Macierz pomyłek:')
cm_xgb = confusion_matrix(y_test, y_pred_xgb_labels)
print(cm_xgb)

# --- Wizualizacja macierzy pomyłek (XGBoost)
macierz_pomylek_XGBoost = True  # True - włączyć / False - wyłączyć. Graficzne wyświetlenie macierzy pomyłek Seaborn (sns.heatmap)

if macierz_pomylek_XGBoost:
    segmenty = ['Lojalni rodzinni', 'Oszczędni aktywni', 'Pasywni', 'VIP' ]
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Purples',
                xticklabels=segmenty, yticklabels=segmenty)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Macierz pomyłek XGBoost')
    plt.tight_layout()
    plt.show()

#  --- Mapping numerów klastrów na nazwy segmentów (etykiety)
segmenty = {
    0: 'Oszczędni aktywni',
    1: 'Lojalni rodzinni',
    2: 'Pasywni',
    3: 'VIP'
}

# --- Przygotowanie DataFrame x: tylko cechy do klasteryzacji, bez braków danych
x = df_scaled[['SUMA_WYDATKÓW', 'RECENCY', 'DOCHOD', 'LICZBA_DZIECI']].dropna().copy()
x['ID'] = df_scaled.loc[x.index, 'ID'].values  # Przywrócenie ID do danych po dropna

# --- Przypisanie numeru klastra i etykiety segmentu każdemu klientowi (po klasteryzacji KMeans)
x['KLASTR'] = kmeans.labels_              # Numer przypisanego klastra
x['SEGMENT'] = x['KLASTR'].map(segmenty)  # Opisowa nazwa segmentu (mapping słownika)

# --- Połączenie segmentacji z oryginalnymi (niestandaryzowanymi) danymi klienta
df_export = df[df['ID'].isin(x['ID'])].copy()  # Tylko klienci bez braków (zaklasteryzowani)
df_export = df_export.merge(
    x[['ID', 'KLASTR', 'SEGMENT']],
    on='ID',
    how='left'
)

# --- Wybór tylko najważniejszych kolumn do eksportu/raportu (do Power BI, dashboardów)
df_export = df_export[[
    'ID', 'SUMA_WYDATKÓW', 'RECENCY', 'DOCHOD', 'LICZBA_DZIECI', 'KLASTR', 'SEGMENT'
]]

# --- Podgląd finalnych danych do eksportu
print()
print('==== DANE w df_export ====')
print(df_export.head())

# --- Eksport do CSV (jeśli eksportowac = True)
eksportowac = False
if eksportowac:
    df_export.to_csv('dane_powerbi_niestandard.csv', index=False, sep=';', encoding='utf-8-sig')
    print('Plik dane_powerbi_niestandard.csv został utworzony.')

# --- Import Streamlit (do tworzenia dashboardu webowego) i ustawienie ciemnego stylu wykresów
import streamlit as st
plt.style.use('dark_background')

# --- Flaga do uruchomienia dashboardu (True – uruchom, False – nie pokazuj)
uruchom_dashboard = False
if uruchom_dashboard:

    # --- Ustawienia i nagłówek dashboardu
    st.title('Dashboard: Segmentacja Klientów')

    # --- Wczytanie danych do dashboardu z pliku CSV
    dane_st = pd.read_csv('dane_powerbi_niestandard.csv', sep=';')

    # --- Podgląd przykładowych danych (pierwsze wiersze)
    st.subheader('Podgląd danych')
    st.dataframe(dane_st.head())

    # --- Wykres: średnie wydatki w każdym segmencie
    st.subheader('Średnie wydatki wg segmentu')
    srednie_wydatki = dane_st.groupby('SEGMENT')['SUMA_WYDATKÓW'].mean()

    # Dostosowanie kolorystyki pod ciemne tło Streamlit
    fig1, ax1 = plt.subplots()
    ax1.bar(srednie_wydatki.index, srednie_wydatki.values, color='#cfcfcf')
    streamlit_tlo = '#0e1117'
    tekst_kolor = '#cfcfcf'
    ax1.set_facecolor(streamlit_tlo)
    fig1.patch.set_facecolor(streamlit_tlo)
    ax1.tick_params(colors=tekst_kolor)
    ax1.spines['bottom'].set_color(tekst_kolor)
    ax1.spines['left'].set_color(tekst_kolor)
    ax1.set_xlabel('Segment', color=tekst_kolor)
    ax1.set_ylabel('Średnie wydatki', color=tekst_kolor)
    ax1.set_title('Średnie wydatki klientów wg segmentu', color=tekst_kolor)
    for i, v in enumerate(srednie_wydatki):
        ax1.text(i, v / 2, f'{v:.0f}', ha='center', va='center', color=streamlit_tlo, fontsize=10, fontweight='bold')

    st.pyplot(fig1)

    # --- Wykres: średni dochód w każdym segmencie
    st.subheader('Średni dochód wg segmentu')
    srednie_dochod = dane_st.groupby('SEGMENT')['DOCHOD'].mean()
    fig2, ax2 = plt.subplots()
    ax2.bar(srednie_dochod.index, srednie_dochod.values)
    ax2.set_xlabel('Segment')
    ax2.set_ylabel('Średni dochód')
    ax2.set_title('Średni dochód klientów wg segmentu')
    st.pyplot(fig2)

    # --- Wykres: średnia liczba dzieci w każdym segmencie
    st.subheader('Średnia liczba dzieci wg segmentu')
    srednie_dzieci = dane_st.groupby('SEGMENT')['LICZBA_DZIECI'].mean()
    fig3, ax3 = plt.subplots()
    ax3.bar(srednie_dzieci.index, srednie_dzieci.values)
    ax3.set_xlabel('Segment')
    ax3.set_ylabel('Średnia liczba dzieci')
    ax3.set_title('Średnia liczba dzieci wg segmentu')
    st.pyplot(fig3)

    # --- Wykres: średnia recency (liczba dni od ostatniego zakupu) w każdym segmencie
    st.subheader('Średnia recency wg segmentu')
    srednia_recency = dane_st.groupby('SEGMENT')['RECENCY'].mean()
    fig4, ax4 = plt.subplots()
    ax4.bar(srednia_recency.index, srednia_recency.values)
    ax4.set_xlabel('Segment')
    ax4.set_ylabel('Średnia recency')
    ax4.set_title('Średnia recency klientów wg segmentu')
    st.pyplot(fig4)

    # --- Wykres: liczba klientów w każdym segmencie (dystrybucja)
    st.subheader('Liczba klientów wg segmentu')
    liczba_klientow = dane_st['SEGMENT'].value_counts()

    fig5, ax5 = plt.subplots()
    ax5.bar(liczba_klientow.index, liczba_klientow.values)
    ax5.set_xlabel('Segment')
    ax5.set_ylabel('Liczba klientów')
    ax5.set_title('Liczba klientów wg segmentu')
    st.pyplot(fig5)


connection.close()



