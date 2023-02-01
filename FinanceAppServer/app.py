from flask import Flask, jsonify
import json
import csv
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify(text='Hello, world')

# 버핏 현재보유 상위 10개 가져오기
@app.route('/wb/top10')
def wb_top_10():
    f = open('./regression/NumShares_Buffet.csv')
    reader = list(csv.reader(f))
    f.close()
    top10 = [i for i in reader if i[0] == '2022-09-01'][:10]

    result = []

    for i in top10:
        dic = dict()
        dic["date"] = i[0]
        dic["security"] = i[1]
        result.append(dic)
    # json_str = json.dumps(result, indent=4)

    return jsonify(result = result)

# 브릿지워터 현재보유 상위 10개 가져오기
@app.route('/bw/top10')
def bw_top_10():
    f = open('./regression/NumShares_RayDalio.csv')
    reader = list(csv.reader(f))
    f.close()
    top10 = [i for i in reader if i[0] == '2022-09-01'][:10]

    result = []

    for i in top10:
        dic = dict()
        dic["date"] = i[0]
        dic["security"] = i[1]
        result.append(dic)
    # json_str = json.dumps(result, indent=4)

    return jsonify(result = result)