from nltk.corpus import stopwords
from collections import namedtuple
import itertools

QA = namedtuple('QA', ['query', 'type', 'doc'])

def qa_processing(PATH):
    list = []
    with open(PATH) as f:
        for line1,line2,line3 in itertools.izip_longest(*[f]*3):
            question = line2[:-1]
            query = [word for word in question.split() if word not in stopwords.words('english')]
            type = line2.split()[0]
            if type == "What's": type = "What"
            list.append(QA(query[1:], type, line1.split()[1]))
    return list

train_PATH = 'qadata/train/questions.txt'
temp = qa_processing(train_PATH)
print(temp)
