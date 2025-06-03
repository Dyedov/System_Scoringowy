import oracledb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans
from scipy.ndimage import rotate

pd.set_option('display.max_columns', None)

connection = oracledb.connect(
    user = 'scoring_user',
    password = 'haslo123',
    dsn = 'localhost/XEPDB1'
)

# with connection.cursor() as cursor:
#     cursor.execute('SELECT COUNT(*) FROM customer_data')
#     liczba = cursor.fetchone()[0]
#     print('Liczba wierszy:', liczba)
#
#     cursor.execute('SELECT * FROM customer_data FETCH FIRST 5 ROWS ONLY')
#     for row in cursor:
#         print(row)
wszystkie_widoki = pd.read_sql('SELECT view_name FROM user_views', con=connection)

print('Widoki w bazie danych:')
print(wszystkie_widoki)

zakupy_df = pd.read_sql('SELECT * FROM VW_ZAKUPY', con=connection)
kampanie_df = pd.read_sql('SELECT * FROM VW_KAMPANIE', con=connection)
demografia_df = pd.read_sql('SELECT * FROM VW_DEMOGRAFIA', con=connection)
online_offline_df = pd.read_sql('SELECT * FROM VW_ONLINE_OFFLINE', con=connection)
recency_df = pd.read_sql('SELECT ID, RECENCY FROM CUSTOMER_DATA', con=connection)

# print(zakupy_df.head(11))
# print(kampanie_df.head())
# print(demografia_df.head())
# print(online_offline_df.head())

# wydatki_kolumny = zakupy_df.drop(columns=['ID'])
# korelacje = wydatki_kolumny.corr()
#
# plt.figure(figsize=(10,8))
# sns.heatmap(korelacje, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title("Heatmapa korelacji wydatków")
# plt.tight_layout()
# plt.show()

dane = zakupy_df.merge(demografia_df, on='ID')
dane = dane.merge(kampanie_df, on='ID')
dane = dane.merge(online_offline_df, on='ID')

print(dane.head())

dane['WIEK'] = 2025 - dane['ROK_URODZENIA']
srednie_wg_wieku = dane.groupby('WIEK')[['WYDATKI_WINO', 'WYDATKI_ZLOTO', 'WYDATKI_SLODYCZE']].mean().reset_index()
# srednie_wg_wieku = srednie_wg_wieku.sort_index(ascending=False)
srednie_wg_wieku = srednie_wg_wieku.round(2)

print('\n')
print('Średnie wydatki wg wieku:')
print(srednie_wg_wieku.head())

# dane['DZIECI'] = dane['DZIECI_W_DOMU'] + dane['NASTOLATKI_W_DOMU']
dane['DZIECI'] = dane[['DZIECI_W_DOMU', 'NASTOLATKI_W_DOMU']].max(axis=1)
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

dane = dane[dane['DOCHOD'].notna()]
dane['PRZEDZIAL_DOCHODU'] = (dane['DOCHOD'] // 10000) * 10000
dane = dane[dane['PRZEDZIAL_DOCHODU'] > 0]

srednie_wg_dochodu = (
    dane.groupby('PRZEDZIAL_DOCHODU')[['WYDATKI_WINO','WYDATKI_ZLOTO', 'WYDATKI_SLODYCZE']]
    .mean()
    .round(2)
    .reset_index()
)

print('\nŚrednie wydatki wg przedziału dochodu:')
print(srednie_wg_dochodu.head(10))

print('\n')
# print(list(online_offline_df.columns))
print('Nagłówki kolumn VW_ONLINE_OFFLINE: ')
for kolumny in online_offline_df.columns:
    print(kolumny)

print('\n')
print('Nagłówki kolumn VW_ZAKUPY: ')
for kolumny in zakupy_df.columns:
    print(kolumny)

df = pd.merge(zakupy_df, online_offline_df, on = 'ID')
df['ONLINE'] = (df['ZAKUPY_ONLINE'] > 0).astype(int)
df['OFFLINE'] = ((df['ZAKUPY_STACJONARNE'] > 0) | (df['ZAKUPY_KATALOGOWE'] > 0)).astype(int)

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

# print(list(kampanie_df.columns))
print('\n')
print('Nagłówki kolumn VW_KAMPANIE: ')
for kolumny in kampanie_df.columns:
    print(kolumny)

df['SUMA_WYDATKÓW'] = (
    df['WYDATKI_WINO']
    + df['WYDATKI_OWOCE']
    + df['WYDATKI_MIESO']
    + df['WYDATKI_RYBY']
    + df['WYDATKI_SLODYCZE']
    + df['WYDATKI_ZLOTO']

)

prog_vip = df['SUMA_WYDATKÓW'].quantile(0.90)
df['VIP'] = (df['SUMA_WYDATKÓW'] >= prog_vip).astype(int)

prog_pasywny = df['SUMA_WYDATKÓW'].quantile(0.10)
df['PASYWNY'] = (df['SUMA_WYDATKÓW'] <= prog_pasywny).astype(int)

df = pd.merge(df, kampanie_df[['ID', 'ODPOWIEDZ_NA_KAMPANIE']], on='ID', how='left')
df['PROMOCYJNY'] = df['ODPOWIEDZ_NA_KAMPANIE'].fillna(0).astype(int)
df = pd.merge(df, recency_df, on='ID', how='left')

# print('\nPrzykładowi klienci z etykietami:')
print('\nPodsumowanie końcowe (z kolumną RECENCY):')
print(df[['ID', 'SUMA_WYDATKÓW', 'VIP', 'PASYWNY', 'PROMOCYJNY', 'RECENCY']].head(10))

print('\nrecency_df HEAD:')
print(recency_df.head())
print('\nKolumny recency_df:', recency_df.columns.tolist())

print('\nLiczba VIP:', df['VIP'].sum())
print('Liczba pasywnych:', df['PASYWNY'].sum())
print('Liczba promocyjnych:', df['PROMOCYJNY'].sum())

# print(f'\nLiczba wszystkich klientów: {len(df)}')
# print(f'10% klientów to: {round(len(df) * 0.10)}')

# print('\nNagłówki kolumn VW_ZAKUPY: ')
# for kolumny in zakupy_df.columns:
#     print(kolumny)
# print('Nagłówki kolumn VW_KAMPANIE: ')
# for kolumny in kampanie_df.columns:
#     print(kolumny)
# print('Nagłówki kolumn VW_DEMOGRAFIA: ')
# for kolumny in demografia_df.columns:
#     print(kolumny)
# print('Nagłówki kolumn VW_ONLINE_OFFLINE: ')
# for kolumny in online_offline_df.columns:
#     print(kolumny)
#
# for nazwa, df in {
# "VW_ZAKUPY": zakupy_df,
#     "VW_KAMPANIE": kampanie_df,
#     "VW_DEMOGRAFIA": demografia_df,
#     "VW_ONLINE_OFFLINE": online_offline_df
# }.items():
#     print(f'\nNagłówki kolumn {nazwa}:')
#     print(df.columns.tolist())
#     print()

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

df = pd.merge(zakupy_df, demografia_df, on='ID', how='left')
df = pd.merge(df, online_offline_df, on='ID', how='left')
df = pd.merge(df, kampanie_df[['ID', 'ODPOWIEDZ_NA_KAMPANIE']], on='ID', how='left')
df = pd.merge(df, recency_df, on='ID', how='left')
df['SUMA_WYDATKÓW'] = (
    df['WYDATKI_WINO']
    + df['WYDATKI_OWOCE']
    + df['WYDATKI_MIESO']
    + df['WYDATKI_RYBY']
    + df['WYDATKI_SLODYCZE']
    + df['WYDATKI_ZLOTO']

)
df['LICZBA_DZIECI'] = df[['DZIECI_W_DOMU', 'NASTOLATKI_W_DOMU']].max(axis=1)

print('\nKolumny w df:')
for col in df.columns:
    print(col)

cechy = ['SUMA_WYDATKÓW', 'RECENCY', 'DOCHOD', 'LICZBA_DZIECI']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# print('\nKolumny w df:', df.columns.tolist())
# print('\nKolumny w df:')
# for col in df.columns:
#     print(col)

df_scaled = scaler.fit_transform(df[cechy])
df_scaled = pd.DataFrame(df_scaled, columns=cechy)
df_scaled['ID'] = df['ID'].values

print('\nDane po standaryzacji (pierwsze 5 wierszy):')
print(df_scaled.head())

# try:
#     import sklearn
#     import matplotlib
#     print('Biblioteka sklearn jest zainstalowana: wersja', sklearn.__version__)
#     print('Biblioteka matplotlib jest zainstalowana: wersja', matplotlib.__version__)
# except ImportError as e:
#     print('Brakuje biblioteki:', e.name)

print('\nKolumny w df:')
for col in df.columns:
    print(col)

from sklearn.cluster import KMeans

x = df_scaled[['SUMA_WYDATKÓW', 'RECENCY', 'DOCHOD', 'LICZBA_DZIECI']].dropna()
inercja = []
zakres_klastrow = range(1, 11)

for k in zakres_klastrow:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x)
    inercja.append(kmeans.inertia_)

# plt.figure(figsize=(8, 5))
# plt.plot(zakres_klastrow, inercja, marker='o')
# plt.xlabel('Liczba klastrów (k)')
# plt.ylabel('Inercja')
# plt.title('Metoda łokcia - wybór liczby klastrów')
# plt.grid(True)
# plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(x)
x['KLASTR'] = kmeans.labels_
grupy = x.groupby('KLASTR').mean().reset_index()
print('\nKlasteryzacja x = df_scaled:')
print(grupy)
print()

segmenty = {
    0: 'Oszczędni aktywni',
    1: 'Lojalni rodzinni',
    2: 'Pasywni',
    3: 'VIP'
}

x['SEGMENT'] = x['KLASTR'].map(segmenty)
print(x[['KLASTR', 'SEGMENT']].value_counts().reset_index(name='liczba_klientów'))
print()

podsumowanie_segmentow = x['SEGMENT'].value_counts().reset_index(name='LICZBA KLIENTÓW').rename(columns={'index': 'SEGMENT'})
print(podsumowanie_segmentow)

# segmenty_liczba = x['SEGMENT'].value_counts()
# segmenty_liczba.plot(kind='bar', color='skyblue')
# plt.title('Liczba klientów w segmentach')
# plt.ylabel('Liczba klientów')
# plt.xlabel('Segment')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

print('\n', x.columns)
print('\n', x.head())
print('\n', x.tail())

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

# print('\n---------------------------------')
# for name, val in list(globals().items()):
#     if isinstance(val, pd.DataFrame):
#         print(name)

dane_model_ml = x.copy()
print('\nDANE_MODEL_ML (Nazwy Kolumn):')
for col in dane_model_ml.columns:
    print(col)

print('\n-------------------------------------')
x_cechy = dane_model_ml[['SUMA_WYDATKÓW', 'RECENCY', 'DOCHOD', 'LICZBA_DZIECI']]
y_segment = dane_model_ml['SEGMENT']
print(y_segment.value_counts())

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x_cechy,
    y_segment,
    test_size=0.2,
    random_state=42,
    stratify=y_segment
)

print('\n======')
print('Liczba próbek treningowych:', len(x_train))
print('Liczba próbek testowych:', len(x_test))
print('\nRozkład klas w zbiorze treningowym:')
print(y_train.value_counts(normalize=True))
print('\nRozkład klas w zbiorze testowym:')
print(y_test.value_counts(normalize=True))

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print('\n-------------------------------------------')
print('Dokładność (accuracy):', accuracy_score(y_test, y_pred))
print('\nRaport klasyfikacji:')
print(classification_report(y_test, y_pred))
print('\nMacierz pomyłek:')
print(confusion_matrix(y_test, y_pred))


connection.close()



