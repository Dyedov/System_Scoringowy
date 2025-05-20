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

wydatki_kolumny = zakupy_df.drop(columns=['ID'])
korelacje = wydatki_kolumny.corr()

plt.figure(figsize=(10,8))
sns.heatmap(korelacje, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Heatmapa korelacji wydatków")
plt.tight_layout()
plt.show()

connection.close()



