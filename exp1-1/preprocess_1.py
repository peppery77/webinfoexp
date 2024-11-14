import pandas as pd
import jieba
from gensim.models import KeyedVectors
from collections import defaultdict

# ... model loading code ...
model = KeyedVectors.load_word2vec_format('light_Tencent_AILab_ChineseEmbedding.bin',binary=True)

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
        if word in processed_words:
            continue
            
        # 获取同义词
        synonyms = set([word] + synonyms_find(word, model, threshold))
        
        # 更新已处理词集合
        processed_words.update(synonyms)
        
        # 添加当前词作为代表词
        unique_words.add(word)
    
    return list(unique_words)

# ... 文件读取代码 ...
file_path = 'D:\\2024秋\\web\\webinfoexp\\exp1-1\\selected_book_top_1200_data_tag.csv'
data = pd.read_csv(file_path)
# 批量处理标签
data['Tags'] = data['Tags'].apply(process_tags)
print(data['Tags'])
data['Tags'] = data['Tags'].apply(lambda x: synonyms_merge(x, model, 0.98))
print(data['Tags'])