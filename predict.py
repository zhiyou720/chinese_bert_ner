from bert import Ner

model = Ner("out_base")

_t = '我是一个很好的记录者这大概也是我孤独的由来我又被谁记录着呢到最后我才明白所有的他们拼凑出来就是一个完整的我'
_t = ' '.join(_t)
_o = model.predict(_t)
res = []
for _item in _o:
    res.append(_item['word'])
    if _item['tag'] != 'word':
        res.append(_item['tag'])

print(f'断句前： {_t}')

print('断句后： ' + ''.join(res).replace("#other#", ""))
