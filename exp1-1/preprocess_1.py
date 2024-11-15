import pandas as pd
import jieba
from gensim.models import KeyedVectors
from collections import defaultdict
import math
import json
import re
from boolExecute import *
from indexStorage import *

# ... model loading code ...
model = KeyedVectors.load_word2vec_format('D:\\2024秋\\web\\webinfoexp\\exp1-1\\light_Tencent_AILab_ChineseEmbedding.bin',binary=True)
with open('D:\\2024秋\\web\\webinfoexp\\exp1-1\\hit_stopwords.txt','r',encoding='utf-8') as f:
    stop_words = f.read().splitlines()

def process_tags(tags_str):
    tags_list = eval(tags_str)
    words = []
    for tag in tags_list:
        words.extend(jieba.cut(tag))
    return list(set(words))

# 创建同义词缓存
synonyms_cache = {}

def synonyms_find(word, model, threshold):
    # 如果已经在缓存中，直接返回
    if word in synonyms_cache:
        return synonyms_cache[word]
    
    try:
        synonyms = model.most_similar(word, topn=5)
        result = [synonym[0] for synonym in synonyms if synonym[1] > threshold]
        # 存入缓存
        synonyms_cache[word] = result
        return result
    except KeyError:
        synonyms_cache[word] = []
        return []

def synonyms_merge(words, model, threshold):
    # 使用集合存储最终结果
    unique_words = set()
    processed_words = set()
    
    for word in words:
        if word in processed_words or word in stop_words:
            continue
            
        # 获取同义词
        synonyms = set([word] + synonyms_find(word, model, threshold))
        
        # 更新已处理词集合
        processed_words.update(synonyms)
        
        # 添加当前词作为代表词
        unique_words.add(word)
    
    return list(unique_words)

def add_skip_pointers(postings):
    n = len(postings)
    skip_distance = int(math.sqrt(n))
    skip_pointers = []
    count = 0
    for i in range(0,n):
        if count % skip_distance == 0:
            skip_pointers.append((postings[i],i+skip_distance if i+skip_distance < n else None))
        else:
            skip_pointers.append((postings[i],None))
        count += 1
    return skip_pointers


# ... 文件读取代码 ...
file_path = 'D:\\2024秋\\web\\webinfoexp\\exp1-1\\selected_book_top_1200_data_tag.csv'
data = pd.read_csv(file_path)
# 批量处理标签
data['Tags'] = data['Tags'].apply(process_tags)
print(data['Tags'])
data['Tags'] = data['Tags'].apply(lambda x: synonyms_merge(x, model, 0.85))
print(data['Tags'])

inverted_index = defaultdict(list)
for idx,tags in enumerate(data['Tags']):
    for tag in tags:
        inverted_index[tag].append(idx)

inverted_index_skips = {word:add_skip_pointers(postings) for word,postings in inverted_index.items()}
print(inverted_index_skips)

with open('inverted_index_skips.json','w',encoding='utf-8') as f:
    json.dump(inverted_index_skips,f,ensure_ascii=False,indent=4)


query = "(动作 and 剧情) or (科幻 and not 恐怖)"
result = execute_boolean_query(query,inverted_index_skips)
print(data['Book'].iloc[0])
result_new = [data['Book'].iloc[i] for i in result]
print(type(result_new[0]))
inverted_index_block = {word:block_storage(postings) for word,postings in inverted_index.items()}
with open('inverted_index_block.json','w',encoding='utf-8') as f:
    json.dump(inverted_index_block,f,ensure_ascii=False,indent=4)

inverted_index_front = {word:front_coding(postings) for word,postings in inverted_index.items()}
with open('inverted_index_front.json','w',encoding='utf-8') as f:
    json.dump(inverted_index_front,f,ensure_ascii=False,indent=4)
