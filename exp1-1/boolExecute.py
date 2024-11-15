import re
#intersect 交集
def intersect_with_skips(p1,p2):
    answer = []
    i,j = 0,0
    while i < len(p1) and j < len(p2):
        if p1[i][0] == p2[j][0]:
            answer.append(p1[i][0])
            i = i+1
            j = j+1
        elif p1[i][0] < p2[j][0]:
            if p1[i][1] is not None and p1[i][1] <p2[j][0]:
                i = p1[i][1]
            else:
                i = i+1
        else:
            if p2[j][1] is not None and p2[j][1] < p1[i][0]:
                j = p2[j][1]
            else:
                j = j+1
    return answer

#union 并集
def union_postings(p1,p2):
    answer = []
    i,j = 0,0
    while i < len(p1) and j < len(p2):
        if p1[i][0] == p2[j][0]:
            answer.append(p1[i][0])
            i = i+1
            j = j+1
        elif p1[i][0] < p2[j][0]:
            answer.append(p1[i][0])
            i = i+1
        else:
            answer.append(p2[j][0])
    while i < len(p1):
        answer.append(p1[i][0])
        i = i+1
    while j < len(p2):
        answer.append(p2[j][0])
        j = j+1
    return answer

#not 相反
def not_postings(p1,p2):
    answer = []
    i,j = 0,0
    while i < len(p1):
        if j > len(p2) or p1[i][0] < p2[j][0]:
            answer.append(p1[i][0])
            i = i+1
        elif p1[i][0] == p2[j][0]:
            i = i+1
            j = j+1
        else:
            j = j+1
    return answer


def parse_boolean_query(query,token_sets):
    query = query.replace('AND','and').replace('OR','or').replace('NOT','not')
    try:
        return eval(query,{"__builtins__":None},{"and":intersect_with_skips,"or":union_postings,"not":not_postings})
    except Exception as e:
        print(f"Error parsing query: {e}")
        raise 

def execute_boolean_query(query,inverted_index_skips):
    tokens = re.findall(r'\b\w+\b', query)
    tokens = [token for token in tokens if token.lower() not in {'and', 'or', 'not'} and token in inverted_index_skips.keys()]
    token_sets = {token:set(doc_id for doc_id,_ in inverted_index_skips.get(token,[])) for token in tokens}
    for token in tokens:
        if token in token_sets:
            query = query.replace(token, str(token_sets[token]))
        else:
            query = query.replace(token, "set()")  # 如果词不在索引中，使用空集合
    print(query)
    return parse_boolean_query(query,token_sets)

