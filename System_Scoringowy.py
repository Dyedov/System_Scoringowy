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

connection.close()



