import re
import collections

'''
编辑单词出现概率
'''


def words(text):
    return re.findall('[a-z]+', text.lower())


def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model


Nwords = train(words(open('big.txt').read()))
print(Nwords)

alphabet = 'abcdefghijklmnopqrstuvwxyz'

'''
编辑距离
'''

# 删、颠倒、置换、增操作算一步
def edits1(word):
    n = len(word)
    return set([word[0:i]+word[i+1:] for i in range(n)] +
               [word[0:i]+word[i+1]+word[i]+word[i+2:] for i in range(n-1)] +
               [word[0:i]+c+word[i+1:] for i in range(n) for c in alphabet] +
               [word[0:i]+c+word[i:] for i in range(n) for c in alphabet])


def edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1))


def known(word):
    return set(w for w in word if w in Nwords)


def correct(word):
    candidates = known([word]) or known(edits1(word)) or known(edits2(word)) or [word]
    return max(candidates, key=lambda w: Nwords[w])


print(correct('learw'))