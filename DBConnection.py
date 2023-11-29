import pyodbc


def connection():
    server = 'localhost\SQLEXPRESS'
    database = 'nolek_testobject'  # replace with your database name

    # For pyodbc
    conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'
                          f'SERVER={server};'
                          f'DATABASE={database};'
                          'Trusted_Connection=yes;')

    return conn