##################################################
# Description: Boolean Query and Inverted Index
##################################################

# imports
import os
import sys
import copy
import operator
from collections import Counter

# data structure for dictionary
class dict:
    def __init__(self, term):
        self.term = term
        self.df = 0
        self.post = None
        self.last_post = None
        
    def get_term(self):
        return self.term
    
    def get_df(self):
        return self.df
    
    def get_post(self):
        return self.post
    
    def get_last_post(self):
        return self.last_post
    
    def set_df(self, df):
        self.df = df
    
    def update_df(self):
        self.df = self.df + 1
        
    def add_post(self, post):
        self.post = post
        
    def update_last_post(self, last_post):
        self.last_post = last_post

# data structure for postings
class post:
    def __init__(self, doc_id, doc_size, tf):
        self.doc_id = doc_id
        self.tf = tf
        self.doc_size = doc_size
        self.next = None
        
    def get_doc_id(self):
        return self.doc_id
    
    def get_tf(self):
        return self.tf / self.doc_size
    
    def get_next(self):
        return self.next
    
    def set_doc_id(self, doc_id):
        self.doc_id = doc_id
        
    def set_tf(self, tf):
        self.tf = tf
        
    def set_next(self, next):
        self.next = nex

# read corpus and tokenize each file
def get_docs(path):
    f = open(path)
    lines = f.readlines()
    
    docs = []
    N = 0
    for line in lines:
        line = line.rstrip()
        line = line.replace('\t', ' ')
        tokens = line.split(' ')
        docs.append(tokens)
        N = N + 1
    return docs, N

# read corpus and tokenize each file
def get_queries(path):
    f = open(path)
    lines = f.readlines()

    queries = []
    for line in lines:
        line = line.rstrip()
        queries.append(line.split(" "))
    return queries

# build inverted index table for the given corpus
def build_ii(docs):
    
    ii_table = []
    terms = []
    
    for tokens in docs:
        doc_id = tokens[0]
        tf = Counter(tokens[1:])

        for term in tf.keys():
            if term not in terms:
                ii = dict(term)
                posting = post(doc_id, len(tokens) - 1, tf[term])
                ii.add_post(posting)
                ii.update_last_post(posting)

                ii.update_df()
                terms.append(term)
                ii_table.append(ii) 
            else:
                idx = terms.index(term)
                last_post = ii_table[idx].get_last_post()
                last_post.next = post(doc_id, len(tokens) - 1, tf[term])
                last_post = last_post.next

                ii_table[idx].update_last_post(last_post)
                ii_table[idx].update_df()
                
    return ii_table, terms

# get postings for queries
def get_postings(queries, ii_table, terms):
    terms_docs = []
    terms_tf = []
    terms_df = []
        
    for term in queries:
        idx = terms.index(term)
        posting = ii_table[idx].get_post()
        term_df = ii_table[idx].get_df()
        
        term_docs = []
        term_tf = []
        while(posting is not None):
            term_docs.append(posting.get_doc_id())
            term_tf.append(posting.get_tf())
            posting = posting.next
            
        terms_docs.append(term_docs)
        terms_tf.append(term_tf)
        terms_df.append(term_df)
            
    return terms_docs, terms_tf, terms_df

# DAAT AND
def daat_and(terms_docs, term_df):
    and_docs = []
    size = len(term_df) - 1
    and_cmp = 0
    
    if len(terms_docs) == 1:
        return terms_docs[0], 0
    
    tmp_term_df = copy.deepcopy(term_df)

    index_min = min(range(len(tmp_term_df)), key = tmp_term_df.__getitem__)
    tmp_term_df[index_min] = 100000
    for doc in terms_docs[index_min]:
        and_docs.append(doc)

    while size > 0:
        index_min = min(range(len(tmp_term_df)), key = tmp_term_df.__getitem__)
        tmp_term_df[index_min] = 100000

        i = 0
        j = 0
        tmp = []
        while i < len(and_docs) and j < len(terms_docs[index_min]):
            while i < len(and_docs) and and_docs[i] < terms_docs[index_min][j]:
                and_cmp = and_cmp + 1
                i = i + 1

            if i == len(and_docs):
                break

            if and_docs[i] == terms_docs[index_min][j]:
                tmp.append(and_docs[i])
                and_cmp = and_cmp + 1
                i = i + 1
                j = j + 1

            if i == len(and_docs):
                break

            while j < len(terms_docs[index_min]) and terms_docs[index_min][j] < and_docs[i]:
                and_cmp = and_cmp + 1
                j = j + 1

            if j == len(terms_docs[index_min]):
                break

        and_docs = copy.deepcopy(tmp)
        size = size - 1

    return and_docs, and_cmp

# DAAT OR
def daat_or(terms_docs, term_df):
    or_docs = []
    size = len(term_df) - 1
    or_cmp = 0
    
    if len(terms_docs) == 1:
        return terms_docs[0], 0
    
    tmp_term_df = copy.deepcopy(term_df)

    index_min = min(range(len(tmp_term_df)), key = tmp_term_df.__getitem__)
    tmp_term_df[index_min] = 100000
    for doc in terms_docs[index_min]:
        or_docs.append(doc)

    while size > 0:
        index_min = min(range(len(tmp_term_df)), key = tmp_term_df.__getitem__)
        tmp_term_df[index_min] = 100000

        i = 0
        j = 0
        tmp = []
        while i < len(or_docs) and j < len(terms_docs[index_min]):
            while i < len(or_docs) and or_docs[i] < terms_docs[index_min][j]:
                tmp.append(or_docs[i])
                or_cmp = or_cmp + 1
                i = i + 1

            if i == len(or_docs):
                break

            if or_docs[i] == terms_docs[index_min][j]:
                tmp.append(or_docs[i])
                or_cmp = or_cmp + 1
                i = i + 1
                j = j + 1

            if i == len(or_docs):
                break

            while j < len(terms_docs[index_min]) and terms_docs[index_min][j] < or_docs[i]:
                tmp.append(terms_docs[index_min][j])
                or_cmp = or_cmp + 1
                j = j + 1

            if j == len(terms_docs[index_min]):
                break

        while i < len(or_docs):
            tmp.append(or_docs[i])
            i = i + 1

        while j < len(terms_docs[index_min]):
            tmp.append(terms_docs[index_min][j])
            j = j + 1

        or_docs = copy.deepcopy(tmp)
        size = size - 1

    return or_docs, or_cmp

# calculate IF-IDF scores
def score(target_docs, terms_docs, terms_df, terms_tf, docs_tf_idf, N):
    i = -1
    for term_docs in terms_docs:
        i = i + 1
        j = -1
        for doc_id in term_docs: 
            j = j + 1
            if doc_id not in target_docs:
                continue

            if doc_id in docs_tf_idf:
                docs_tf_idf[doc_id] = docs_tf_idf[doc_id] + terms_tf[i][j] * N / terms_df[i]
            else:
                docs_tf_idf[doc_id] = terms_tf[i][j] * N / terms_df[i]
        
    return docs_tf_idf

# rank the documents according to TF-IDF scores
def rank(docs_tf_idf):
    sorted_tf_idf = sorted(docs_tf_idf.items(), key=operator.itemgetter(1), reverse=True)

    # f = open("tf_idf.txt", "w+")
    # for t in sorted_tf_idf:
    #     f.write(str(t) + "\n")
    # f.close()
    
    sorted_docs = []
    for item in sorted_tf_idf:
        sorted_docs.append(item[0])
    return sorted_docs

# boolean query
def boolean_query(path_input, path_output, ii_table, terms, N):
    queries = get_queries(path_input)
    f = open(path_output, "w+")
    
    for query in queries:
        terms_docs, terms_tf, terms_df = get_postings(query, ii_table, terms)
        
        for i in range(0, len(terms_docs)):
            f.write("GetPostings\n")
            f.write(query[i])
            f.write("\nPostings list: " + ' '.join(terms_docs[i]) + "\n")
        
        and_docs, and_cmp = daat_and(terms_docs, terms_df)
        or_docs, or_cmp = daat_or(terms_docs, terms_df)
        
        if len(and_docs) > 0:
            docs_tf_idf = {}
            docs_tf_idf = score(and_docs, terms_docs, terms_df, terms_tf, docs_tf_idf, N)
            and_tf_idf = rank(docs_tf_idf)
        
        if len(or_docs) > 0:
            docs_tf_idf = {}
            docs_tf_idf = score(or_docs, terms_docs, terms_df, terms_tf, docs_tf_idf, N)
            or_tf_idf = rank(docs_tf_idf)
        
        f.write("DaatAnd\n")
        f.write(' '.join(query) + "\n")
        if len(and_docs) == 0:
            f.write("Results: empty\n")
        else: 
            f.write("Results: " + ' '.join(and_docs) + "\n")
        f.write("Number of documents in results: " + str(len(and_docs)) + "\n")
        f.write("Number of comparisons: " + str(and_cmp) + "\n")
        f.write("TF-IDF\n")
        if len(and_docs) == 0:
            f.write("Results: empty\n")
        else:
            f.write("Results: " + ' '.join(and_tf_idf) + "\n")
        
        f.write("DaatOr\n")
        f.write(' '.join(query) + "\n")
        if len(or_docs) == 0:
            f.write("Results: empty\n")
        else: 
            f.write("Results: " + ' '.join(or_docs) + "\n")
        f.write("Number of documents in results: " + str(len(or_docs)) + "\n")
        f.write("Number of comparisons: " + str(or_cmp) + "\n")
        f.write("TF-IDF\n")
        if len(or_docs) == 0:
            f.write("Results: empty\n")
        else:
            f.write("Results: " + ' '.join(or_tf_idf) + "\n")
        
        if query != queries[len(queries)-1]:
            f.write("\n")
        
    f.close()

# main() function
def main():
	args = sys.argv
	path_corpus = args[1]
	path_output = args[2]
	path_input = args[3]

	docs, N = get_docs(path_corpus)
	ii_table, terms = build_ii(docs)
	boolean_query(path_input, path_output, ii_table, terms, N)

	

# program starts here
if __name__== "__main__":
	main()