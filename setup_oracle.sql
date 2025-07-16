-- Połączenie z lokalną bazą danych Oracle
-- localhost/XEPDB1

-- Tworzenie użytkownika i nadanie uprawnień
CREATE USER scoring_user IDENTIFIED BY haslo123;
GRANT CONNECT, RESOURCE TO scoring_user;

-- Przyznanie użytkownikowi uprawnienia do przestrzeni tabel USERS
ALTER USER scoring_user QUOTA UNLIMITED ON USERS;

-- Przyznanie użytkownikowi uprawnienia CREATE VIEW
GRANT CREATE VIEW TO scoring_user;

-- Tworzenie widoków

CREATE OR REPLACE VIEW vw_zakupy AS
SELECT
  id,
  mntwines AS wydatki_wino,
  mntfruits AS wydatki_owoce,
  mntmeatproducts AS wydatki_mieso,
  mntfishproducts AS wydatki_ryby,
  mntsweetproducts AS wydatki_slodycze,
  mntgoldprods AS wydatki_zloto
FROM customer_data;

CREATE OR REPLACE VIEW vw_kampanie AS
SELECT
  id,
  acceptedcmp1 AS kampania_1,
  acceptedcmp2 AS kampania_2,
  acceptedcmp3 AS kampania_3,
  acceptedcmp4 AS kampania_4,
  acceptedcmp5 AS kampania_5,
  response AS odpowiedz_na_kampanie
FROM customer_data;

CREATE OR REPLACE VIEW vw_demografia AS
SELECT
  id,
  year_birth AS rok_urodzenia,
  education AS wyksztalcenie,
  marital_status AS stan_cywilny,
  income AS dochod,
  kidhome AS dzieci_w_domu,
  teenhome AS nastolatki_w_domu
FROM customer_data;

CREATE OR REPLACE VIEW vw_online_offline AS
SELECT
  id,
  numwebpurchases AS zakupy_online,
  numstorepurchases AS zakupy_stacjonarne,
  numcatalogpurchases AS zakupy_katalogowe,
  numwebvisitsmonth AS wizyty_na_stronie_miesiecznie
FROM customer_data;

-- Sprawdzenie utworzonych widoków
SELECT * FROM user_views;

-- Pobranie recency
SELECT ID, RECENCY FROM customer_data;

-- Utworzenie widoku VW_RECENCY
CREATE OR REPLACE VIEW vw_recency AS 
SELECT ID, RECENCY
FROM customer_data;
