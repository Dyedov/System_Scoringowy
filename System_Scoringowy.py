import oracledb

connection = oracledb.connect(
    user = 'scoring_user',
    password = 'haslo123',
    dsn = 'localhost/XEPDB1'
)

with connection.cursor() as cursor:
    cursor.execute('SELECT COUNT(*) FROM customer_data')
    liczba = cursor.fetchone()[0]
    print('Liczba wierszy:', liczba)

    cursor.execute('SELECT * FROM customer_data FETCH FIRST 5 ROWS ONLY')
    for row in cursor:
        print(row)

connection.close()