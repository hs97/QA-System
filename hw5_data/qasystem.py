from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer, sent_tokenize
from nltk import ngrams, pos_tag
from collections import namedtuple, Counter
from scipy import spatial
from lxml import etree
from itertools import izip_longest
from nltk.tag import StanfordNERTagger


QA = namedtuple('QA', ['question', 'query', 'type', 'qid'])
QC = {'Who':['PERSON','ORGANIZATION'], 'Where':['LOCATION'], 'What':['NN', 'NNP'], 'When':['DATE', 'TIME'], 'How many':['CD'], 'Name':['NNP']}
st = StanfordNERTagger('english.muc.7class.distsim.crf.ser.gz',
		        'stanford-ner.jar',
			encoding='utf-8')
passage = namedtuple('passage', ['ngram', 'pos', 'ne'])

# sets lowercase, removes stop words and punctuation
def preprocess(sent):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sent)
    return [t for t in tokens if not t in set(stopwords.words('english'))]

# creates a list of tuples with question, query, question type, and qid
def qa_processing(PATH):
    questions = []
    with open(PATH) as f:
        for line1,line2,line3 in izip_longest(*[f]*3):
            question = line2
            query = preprocess(question)[1:]
            if line2.split()[0:1] != 'How many':
                type = line2.split()[0]
            else:
                type = 'How many'
            if type == "Can": type = "Name"
            if type in ["What's", 'For', 'Tell', 'At']: type = "What"
            questions.append(QA(question, query, type, line1.split()[1]))
    return questions

# calculates cosine similarity of two lists - can change similarity measurement later
def cos_similarity(ngrams, query):
    wordbag = query
    vector_a = [int(w in ngrams) for w in wordbag]
    vector_b = [int(w in query) for w in wordbag]
    if 1 not in vector_a:
        return 0
    else:
        return 1 - spatial.distance.cosine(vector_a, vector_b)

def similarity(passage, query, n):
    cos_sim = {ngram:cos_similarity(list(ngram), query) for sent in passage for ngram in ngrams(sent, n)}
    ## Will take NE and POS into accounting for similarity
    return cos_sim

def passage_retrieval(entry):
    tree = etree.parse('topdocs/test/top_docs.' + entry.qid, parser=etree.HTMLParser(remove_comments=True))
    root = tree.getroot()
    top = {}
    target = ''
    for r in root[0][0]:
        if r.find('text') is not None:
            # sometimes, the text is nested within <P> and </P> within <TEXT> and </TEXT>
            if r.find('text').find('p') is not None: paragraph = ''.join([p.text for p in r.find('text')])
            else: paragraph = r.find('text').text
        # Will use 'passage' named tuple to carry NE and POS 
        paragraph = sent_tokenize(str(paragraph))
        paragraph = [preprocess(sent) for sent in paragraph]
        top.update(similarity(paragraph, entry.query, 15))
    top = sorted(top.items(), key = lambda x:-x[1])[:20]
    return top

def answer_processing(top, entry): 
#    print(top)
    if entry.type == 'How':
        return [' '.join(tup[0]) for tup in top] 
    if entry.type in ['Who', 'Where', 'When']:  
        top_ten = []
        for ngram in top:
            #classified text
            ct = st.tag(ngram[0])
            prev_string = ''
            for i in range(len(ct) - 1):
                if ct[i][1] in QC[entry.type] and len(top_ten) < 10:
                    if prev_string == '':
                        if str(ct[i][0]) not in top_ten:
                            top_ten = top_ten + [str(ct[i][0])]
                        prev_string = str(ct[i][0])
                    else:
                        if [prev_string + ' ' + str(ct[i][0])] not in top_ten:                        
                            top_ten = top_ten + [prev_string + ' ' + str(ct[i][0])]
                            top_ten.remove(prev_string)
                        prev_string = prev_string + ' ' + str(ct[i][0])
                        if ct[i + 1][1] not in QC[entry.type]:
                            prev_string = ''
                top_ten = list(set(top_ten))
        return top_ten
    else: 
        top_ten = []
        for ngram in top:
            #classified text
            ct = pos_tag(ngram[0])
            prev_string = ''
            for i in range(len(ct) - 1):
                if ct[i][1] in QC[entry.type] and len(top_ten) < 10 and ct[i][0] not in entry.query:
                    if prev_string == '':
                        if ct[i][0] not in top_ten:
                            top_ten = top_ten + [ct[i][0]]
                        prev_string = ct[i][0]
                    else:
                        if (prev_string + ' ' + ct[i][0]) not in top_ten:
                            top_ten = top_ten + [prev_string + ' ' + ct[i][0]]
                            top_ten.remove(prev_string)
                        prev_string = prev_string + ' ' + ct[i][0]
                        if ct[i + 1][1] not in QC[entry.type]:
                            prev_string = ''
        return top_ten 
         
        
train_qPATH = 'qadata/test/questions.txt'

temp = qa_processing(train_qPATH)
for t in temp:
    top = passage_retrieval(t)
    print("qid " + t.qid)
    answer = answer_processing(top, t)
    for a in answer: print(a)

