import re

import jieba
from jieba import cut
from jieba.posseg import cut as p_cut
from pandas import DataFrame
from wordcloud import WordCloud
with open('./data/stopwords.txt', encoding='utf-8') as f:
    stopwords = [re.sub(r'\n', '', word) for word in f.readlines()]
jieba.load_userdict('./data/dict.txt')

txt = """MoneyDJ新聞 2018-01-03 07:23:31 記者 陳苓 報導半導體產業協會(
SIA)2日公布，2017年11月份全球半導體銷售額為377億美元。和前月相比，11月銷售額提升1.6%。和去年同期相比，大增21.5%。SIA總裁兼執行長John 
Neuffer聲明稿指出，全球半導體業11月再創新猷，單月銷售額又破空前新高；2017年的年度銷售應會達到4,
000億美元，創下首例。記憶體產品持續帶動全球市場成長，不過其他各類半導體的銷售也全數出現年增和月增。11月各地區市場皆呈現成長，尤以美洲漲幅最大。和去年同期相比，美洲銷售增40.2%、歐洲增18.8%、中國增18.5
%、亞太/其他地區增16.2%、日本增10.6%。 (詳細數據)費城半導體指數2日上漲2.77%、收1,287.70點。"""
txt = re.sub(r'\W|\d|\s|[a-zA-Z]', '', txt)
tokens = p_cut(txt)

pos_list = [[k, v] for k, v in tokens]
df = DataFrame(pos_list, columns=['token', 'pos'])
df.to_csv('./data/pos.csv', index=False)

tokens = cut(txt)
seg_lsit = []
for word in tokens:
    if word not in stopwords:
        seg_lsit.append(word)

cloud = WordCloud(background_color="white").generate(' '.join(seg_lsit))
cloud.to_file(r'./data/cloud.png')

from gensim.models.fasttext import FastText

model = FastText(seg_lsit, size=10)
vec = model.wv['吃飯']
print(vec)