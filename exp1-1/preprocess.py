import pandas as pd
import jieba
from gensim.models import KeyedVectors
from collections import defaultdict
model = KeyedVectors.load_word2vec_format('light_Tencent_AILab_ChineseEmbedding.bin',binary=True)

def process_tags(tags_str):
    tags_list = eval(tags_str)
    words=[]
    for tag in tags_list:
        words.extend(jieba.cut(tag))
    return list(set(words))


def synonms_find(word,model,threhold):
    try:
        synonms = model.most_similar(word,topn = 5)
        return [synonm[0] for synonm in synonms if synonm[1] > threhold]
    except KeyError:
        return []
    
def synonyms_merge(words,model,threhold):
    synonms = []
    synonms_dict = {}
    for word in words:
        synonms = synonms_find(word,model,threhold)
        if word not in synonms_dict.values():
            synonms_dict[word] = set([word] + synonms)
        else:
            continue
    return list(set(synonms_dict.keys()))

file_path = 'D:\\2024秋\\web\\webinfoexp\\exp1-1\\selected_book_top_1200_data_tag.csv'
data = pd.read_csv(file_path)
print(type(data))
data['Tags'] = data['Tags'].apply(process_tags)
# print(data['Book'].iloc[0])
data['Tags'] = data['Tags'].apply(lambda x: synonyms_merge(x,model,0.98))

