from flask import Flask, request, jsonify
from flask_cors import CORS

from bert import Ner

app = Flask(__name__)
CORS(app)

model = Ner("out_base")


@app.route("/predict", methods=['POST'])
def predict():
    text = request.json["text"]
    try:
        out = model.predict(text)
        return jsonify({"result": out})
    except Exception as e:
        print(e)
        return jsonify({"result": "Model Failed"})


if __name__ == "__main__":
    # app.run('0.0.0.0', port=8000)
    _t = '我是一个很好的记录者这大概也是我孤独的由来我又被谁记录着呢到最后我才明白所有的他们拼凑出来就是一个完整的我'
    _t = ' '.join(_t)
    _o = model.predict(_t)
    res = []
    for _item in _o:
        res.append(_item['word'])
        if _item['tag'] != 'word':
            res.append(_item['tag'])

    print('断句前： {}'.format(_t))

    print('断句后： ' + ''.join(res))
