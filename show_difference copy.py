import mysql.connector
import json
from matplotlib import pyplot as plt
from util.config_util import DB_HOST

if __name__ == "__main__":
    db = mysql.connector.connect(
        host="192.168.0.113",
        # host=DB_HOST,
        user="root",
        password="123456",
        database="ayesha",
        auth_plugin='mysql_native_password'
        )

    cursor = db.cursor()
    cursor.execute('''
select a.score, b.score, a.item_id
from (
         select avg(score) score, a.item_id
         from llm_answer a
                  inner join difference_item di on a.item_id = di.item_id
         where a.llm_name = 'ChatGLM-6B'
         group by a.item_id) a
         inner join (
    select avg(score) score, a.item_id
    from llm_answer a
             inner join difference_item di on a.item_id = di.item_id
    where a.llm_name = 'ChatGLM-6B-difference'
    group by a.item_id) b on a.item_id = b.item_id
''')
    results = cursor.fetchall()
    cursor.close()
    db.close()
    y1 = []
    y2 = []
    x = []
    for i in range(len(results)):
        original_score = results[i][0]
        y1.append(original_score)
        hint_score = results[i][1]
        y2.append(hint_score)
        item_index = i+1
        x.append(item_index)

    plt.scatter(x, y1, color = 'hotpink')
    # plt.scatter(x, y2, color = '#88c999')

    plt.xlabel("试题不确定度")
    plt.ylabel("试题数量")

    plt.rcParams['font.sans-serif']=['Songti SC']
    plt.rcParams['axes.unicode_minus']=False
    # function to show the plot
    plt.show()

    # db = mysql.connector.connect(
    #     # host="192.168.0.102",
    #     host=DB_HOST,
    #     user="root",
    #     password="123456",
    #     database="ayesha",
    #     auth_plugin='mysql_native_password'
    #     )
    # cursor = db.cursor()
    # for k,v in item_id_difference_map.items():
    #     if v >= 4.0:
    #         cursor.execute("""
    #     INSERT INTO `ayesha`.`difference_item`
    #     (`item_id`)
    #     VALUES
    #     (%(item_id)s);
    #     """ ,{
    #         "item_id": k
    #         })
    # cursor.close()
    # db.commit()
    # db.close()
    # print(item_id_difference_map)
