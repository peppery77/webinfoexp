def block_storage(postings):
    block_size = 128
    blocks = []
    for i in range(0,len(postings),block_size):
        block = postings[i: i + block_size]
        base_id = block[0]
        offsets = [doc_id - base_id for doc_id in postings]
        blocks.append((base_id,offsets))
    return blocks

def front_coding(postings):
    encoded = []
    prev_str = str(postings[0])
    encoded.append((0,prev_str))
    for i in range(1,len(postings)):
        current_str = str(postings[i])
        prefix_len = 0
        while prefix_len < len(prev_str) and prefix_len < len(current_str) and prev_str[prefix_len] == current_str[prefix_len]:
            prefix_len = prefix_len+1
        suffix = current_str[prefix_len:]
        encoded.append((prefix_len,suffix))
        prev_str = current_str
    return encoded

