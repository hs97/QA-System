from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk import ngrams
from collections import namedtuple, Counter
from scipy import spatial
import xml.etree.ElementTree as ElementTree
import itertools, sys

QA = namedtuple('QA', ['question', 'query', 'type', 'qid'])

# sets lowercase, removes stop words and punctuation
def preprocess(sent):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sent.lower())
    return [t for t in tokens if not t in set(stopwords.words('english'))]

# creates a list of tuples with question, query, question type, and qid
def qa_processing(PATH):
    questions = []
    with open(PATH) as f:
        for line1,line2,line3 in itertools.zip_longest(*[f]*3):
            question = line2[:-1]
            query = preprocess(question)
            type = line2.split()[0]
            if type == "Can": type = "Name"
            if type in ["What's", 'For', 'Tell', 'At']: type = "What"
            questions.append(QA(question, query, type, line1.split()[1]))
    return questions

# calculates cosine similarity of two lists - can change similarity measurement later
def cos_similarity(a,b):
    wordbag = set(a+b)
    vector_a = [int(w in a) for w in wordbag]
    vector_b = [int(w in b) for w in wordbag]
    return 1 - spatial.distance.cosine(vector_a, vector_b)

def passage_retrieval(entry):
    with open('topdocs/train/top_docs.' + entry.qid, 'r') as f: xml = f.read()
    xml = '<ROOT>' + xml + '</ROOT>'
    root = ElementTree.fromstring(xml)
    i = 0
    for r in root:
        print(r.find('DOCNO').text.strip())
        paragraph = r.find('TEXT').text
        tengrams = ngrams(preprocess(paragraph), 10)
        for t in tengrams: 
            print(cos_similarity(list(t), entry.query))
        if i == 3: sys.exit()
        i += 1
        
train_qPATH = 'qadata/train/questions.txt'
train_dPATH = 'qadata/train/relevant_docs.txt'
temp = qa_processing(train_qPATH)
passage_retrieval(temp[0])
