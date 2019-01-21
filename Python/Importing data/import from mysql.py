# Import tables in mysql to dataframe
import mysql.connector
cnx = mysql.connector.connect(user='root', password='1234',
                              host='localhost',
                              database='test')
cursor = cnx.cursor()
cursor.execute('SELECT * FROM dim_member')
colname = cursor.column_names
a = cursor.fetchall()
import pandas as pd
b = pd.DataFrame(a)
b.columns = colname
print(b)
cursor.close()