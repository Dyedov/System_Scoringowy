import oracledb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

print('\nPrzykładowi klienci z etykietami:')
print(df[['ID', 'SUMA_WYDATKÓW', 'VIP', 'PASYWNY', 'PROMOCYJNY']].head(10))

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


connection.close()



