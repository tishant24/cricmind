import pymysql
try:
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='tumhara_password_yahan',
        port=3306
    )
    print('✅ MySQL Connected!')
    conn.close()
except Exception as e:
    print(f'❌ Error: {e}')