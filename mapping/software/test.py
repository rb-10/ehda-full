import sqlite3, os
con = sqlite3.connect(r"C:/Users/HV/Desktop/bruno_work/save_electrospray/data.db")
cols = [row[1] for row in con.execute("PRAGMA table_info(measurements)").fetchall()]
print(cols)
con.close()