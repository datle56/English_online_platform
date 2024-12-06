import os

def load_vocab(dir):
    vocab_path = os.path.join(dir, 'labels.txt')
    vocab = open(vocab_path).read().replace('@@PADDING@@', '').replace('@@UNKNOWN@@', '').rstrip().split('\n')
    # vocab_d = open(dir + 'd_tags.txt').read().rstrip().replace('@@PADDING@@', '<PAD>').replace('@@UNKNOWN@@', '<OOV>').split('\n')
    label2id = {'<OOV>':0, '$KEEP':1}
    d_label2id = {'$CORRECT':0, '$INCORRECT':1, '<PAD>':2}
    idx = len(label2id)
    for v in vocab:
        if v not in label2id:
            label2id[v] = idx
            idx += 1
    label2id['<PAD>'] = idx
    return label2id, d_label2id
        