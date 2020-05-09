# -*-coding:utf-8
import re

import jieba
from jieba import cut, load_userdict
from jieba.posseg import cut as p_cut
from pandas import DataFrame

from collections import Counter
from wordcloud import WordCloud
from gensim.models.fasttext import FastText
# xxxx = jieba.cut()
# msyh.ttf

with open('./data/stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = [re.sub(r'\n', '', word) for word in f.readlines()]
    # stopwords = f.readlines()

# print(stopwords)
load_userdict('./data/dict.txt')
txt = """MoneyDJ新聞 2018-01-03 07:23:31 記者 陳苓 報導半導體產業協會(
SIA)2日公布，2017年11月份全球半導體銷售額為377億美元。和前月相比，11月銷售額提升1.6%。和去年同期相比，大增21.5%。SIA總裁兼執行長John 
Neuffer聲明稿指出，全球半導體業11月再創新猷，單月銷售額又破空前新高；2017年的年度銷售應會達到4,
000億美元，創下首例。記憶體產品持續帶動全球市場成長，不過其他各類半導體的銷售也全數出現年增和月增。11月各地區市場皆呈現成長，尤以美洲漲幅最大。和去年同期相比，美洲銷售增40.2%、歐洲增18.8%、中國增18.5
%、亞太/其他地區增16.2%、日本增10.6%。 (詳細數據)費城半導體指數2日上漲2.77%、收1,287.70點。"""

txt = re.sub(r'\W|\s|\d|[a-zA-Z]', '', txt)

tokens = cut(txt)

output = []

for token in tokens:
    if token not in stopwords:
        output.append(token)

model = FastText(output, min_count=1, workers=8, size=3)
get_vector = model.wv['半導體']
print(get_vector)
# for token in tokens:
#     if token not in stopwords:
#         output.append(token)

# wc = WordCloud(font_path=r'./data/msyh.ttf', background_color='white').generate(' '.join(output))

# wc.to_file(r'./data/wc.png')

# print(output)

# count = Counter(output)
# print(count)

# for k in count.items():
#     print(k)
# pos_tokens = p_cut(txt)

# pos_ = []

# for word, pos in pos_tokens:
#     pos_.append([word, pos])

# pos_df = DataFrame(pos_, columns=['word', 'pos'])

# pos_df.to_csv('./data/pos.csv', index=False)
