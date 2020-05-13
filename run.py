#  -*-coding:utf-8-*-
import re
from collections import Counter

import jieba
from gensim.models.fasttext import FastText
from jieba import cut, load_userdict
from jieba.posseg import cut as p_cut
from pandas import DataFrame
from wordcloud import WordCloud


def pos(txt):
    """
    有詞性的斷詞，並且存成csv

    Args:
        txt (str): 已清理的文本
    """

    pos_tokens = p_cut(txt)
    pos_ = []

    for word, pos in pos_tokens:
        pos_.append([word, pos])

    pos_df = DataFrame(pos_, columns=['word', 'pos'])
    pos_df.to_csv('./data/pos.csv', index=False)


def seg(txt, stopwords):
    """
    純斷詞並篩掉停傭詞

    Args:
        txt (str): 已清理文本
        stopwords (list): 載入的停用詞

    Returns:
        output (list): 斷詞後的tokens
    """
    tokens = cut(txt)
    output = []
    for token in tokens:
        if token not in stopwords:
            output.append(token)
    return output


def word_cloud(tokens):
    """
    製作文字雲

    Args:
        tokens (list): 斷詞後的tokens
    """
    wc = WordCloud(font_path='./data/msyh.ttf',
                   background_color='white').generate(' '.join(tokens))
    wc.to_file('./data/wc.png')


def word_embedding(tokens):
    """
    訓練詞向量並儲存

    Args:
        tokens (list): 斷詞的tokens
    """
    # input must be 2D even only one data
    # tokens = [['xx', 'xxx']]
    model = FastText([tokens], min_count=1, size=3)
    get_vector = model.wv['半導體']
    model.save('./data/wb.model')
    print(get_vector)
    print(model.wv.vocab)


def main():

    txt = """MoneyDJ新聞 2018-01-03 07:23:31 記者 陳苓 報導半導體產業協會(
    SIA)2日公布，2017年11月份全球半導體銷售額為377億美元。和前月相比，11月銷售額提升1.6%。和去年同期相比，大增21.5%。SIA總裁兼執行長John 
    Neuffer聲明稿指出，全球半導體業11月再創新猷，單月銷售額又破空前新高；2017年的年度銷售應會達到4,
    000億美元，創下首例。記憶體產品持續帶動全球市場成長，不過其他各類半導體的銷售也全數出現年增和月增。11月各地區市場皆呈現成長，尤以美洲漲幅最大。和去年同期相比，美洲銷售增40.2%、歐洲增18.8%、中國增18.5
    %、亞太/其他地區增16.2%、日本增10.6%。 (詳細數據)費城半導體指數2日上漲2.77%、收1,287.70點。"""

    # 載入文本
    load_userdict('./data/dict.txt')
    # 載入停用詞
    with open('./data/stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = [re.sub(r'\n', '', word) for word in f.readlines()]

    # 文字清理(只留中文)
    txt = re.sub(r'\W|\s|\d|[a-zA-Z]', '', txt)

    # 執行pos
    pos(txt)

    # 執行seg
    tokens = seg(txt, stopwords)

    # 計算tf
    count = Counter(tokens)
    # print(count)

    # 執行word cloud
    word_cloud(tokens)

    # 執行word embedding
    word_embedding(tokens)


if __name__ == "__main__":
    main()
