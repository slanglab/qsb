# coding: utf-8
import pdb

from bottom_up.all import EasyAllenNLP

m = EasyAllenNLP()
paper_json = {"tokens": [{'word': "hi"}, {"word": "bye"}]}
print(m.predict_proba(paper_json)) 
