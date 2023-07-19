import mysql.connector
import json

from bs4 import BeautifulSoup
import jieba

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="123456",
  database="ayesha"
)

mycursor = mydb.cursor()

mycursor.execute("SELECT * FROM item_doc limit 100")

myresult = mycursor.fetchall()

for x in myresult:
  item = json.loads(x[1])
  raw_html = item['bundler']['content']
  print(x[2])
#   print("raw_html: ", raw_html)
  cleantext = BeautifulSoup(raw_html, "html.parser").text
#   print("cleantext: ", cleantext)
  seg_list = jieba.cut(cleantext) 
  print("Paddle Mode: " + '/'.join(list(seg_list)))

mycursor.close()