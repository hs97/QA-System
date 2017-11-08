from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk import ngrams, pos_tag
from collections import namedtuple, Counter
from scipy import spatial
from lxml import etree
from itertools import izip_longest
from nltk.tag import StanfordNERTagger


QA = namedtuple('QA', ['question', 'query', 'type', 'qid'])
QC = {'Who':['PERSON','ORGANIZATION'], 'Where':['LOCATION'], 'What':['NN', 'NNP'], 'When':['DATE', 'TIME'], 'How many':['CD'], 'Name':['NN', 'NNP']}
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
    cos_sim = {ngram:cos_similarity(list(ngram), query) for ngram in ngrams(passage, n)}
    ## Will take NE and POS into accounting for similarity
    return cos_sim

def passage_retrieval(entry):
    tree = etree.parse('topdocs/train/top_docs.' + entry.qid, parser=etree.HTMLParser(remove_comments=True))
    root = tree.getroot()
    top = {}
    target = ''
    for r in root[0][0]:
        if r.find('text') is not None:
            # sometimes, the text is nested within <P> and </P> within <TEXT> and </TEXT>
            if r.find('text').find('p') is not None: paragraph = ''.join([p.text for p in r.find('text')])
            else: paragraph = r.find('text').text
        # Will use 'passage' named tuple to carry NE and POS 
        paragraph = preprocess(str(paragraph))
        top.update(similarity(paragraph, entry.query, 20))
    top = sorted(top.items(), key = lambda x:-x[1])[:10]
    return top

def answer_processing(top, entry): 
#    print(top)
    if entry.type == 'How':
        return top 
    if entry.type in ['Who', 'Where', 'When']:  
        classified_text = [st.tag(ngram[0]) for ngram in top]
        classified_text = [[str(tag[0]) for tag in ngram if str(tag[1]) in QC[entry.type] if str(tag[0]) not in entry.query] for ngram in classified_text]
        return classified_text
    
    else: 
        tagged_text = [pos_tag(ngram[0]) for ngram in top]
        tagged_text = [[tag[0] for tag in ngram if tag[1] in QC[entry.type] if tag[0] not in entry.query] for ngram in tagged_text]
        return tagged_text
        
         
        
train_qPATH = 'qadata/train/questions.txt'
train_dPATH = 'qadata/train/relevant_docs.txt'

temp = qa_processing(train_qPATH)
for t in temp:
    top = passage_retrieval(t)
    print("qid " + t.qid)
    answer = answer_processing(top, t)
    for a in answer: print(' '.join(a))
#    print(t.query)
#top = passage_retrieval(temp[14])
#print(answer_processing(top, temp[14]))
#print(temp[14].question)
