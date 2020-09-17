import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import time
import pickle

features = ['TF(Term frequency) of body', 'TF of anchor', 'TF of title', 'TF of URL', 'TF of whole document', 'IDF(Inverse document frequency) of body', 
            'IDF of anchor', 'IDF of title', 'IDF of URL', 'IDF of whole document', 'TF*IDF of body', 'TF*IDF of anchor',
            'TF*IDF oftitle', 'TF*IDF of URL', 'TF*IDF of whole document', 'DL(Document length) of body' , 'DL of anchor', 'DL of title', 'DL of URL', 
            'DL of whole document', 'BM25 of body', 'BM25 of anchor', 'BM25 of title', 'BM25 of URL','BM25 of whole document', 
            'LMIR.ABS of body', 'LMIR.ABS of anchor', 'LMIR.ABS of title', 'LMIR.ABS of URL', 'LMIR.ABS of whole document', 'LMIR.DIR of body', 'LMIR.DIR of anchor',
            'LMIR.DIR of title', 'LMIR.DIR of URL', 'LMIR.DIR of whole document', 'LMIR.JM of body', 'LMIR.JM of anchor', 'LMIR.JM of title', 'LMIR.JM of URL', 
            'LMIR.JM of whole document', 'PageRank', 'Inlink number', 'Outlink number', 'Number of slash in URL', 'Length of URL', 'Number of child']

fold_num = 5

train = pd.read_csv('./data/MQ2008-list/I{}.txt'.format(fold_num),  sep=' ', header=None)

train = train.drop([48, 49, 51, 52,53, 54, 55, 56], axis=1)

columns = ['rank_num', 'query']

for i in range(len(features)):
    columns.append(features[i])

columns.extend(['doc_id'])
train.columns = columns

result = []
threshold = 204
start = time.time()

query_counter = 0
for query_key, group_df_query in train.groupby(['query']):
    print(query_counter)
    q_id = query_key.split(':')[1]
    print('query', q_id)
    query_counter +=1
    tree = ET.parse('./data/MetaFeature/query-{}.xml'.format(q_id))
    q_meta = tree.getroot()
    q_value = []
    for i in range(1, int(q_meta[3][0].text)  + 1):
        q_value.append(q_meta[3][i][1].text)
    result_dic = {}
    result_dic['query'] = ' '.join(q_value) 
    result_dic['docs'] = []
    counter = 0
    for doc_key, group_df_query_doc in group_df_query.groupby(['doc_id']):
        if (counter < threshold):
            result_doc_dic = {}
            result_doc_dic['doc_key'] = group_df_query_doc.values[0][-1]
            for f_counter in range(2, 47):
                result_doc_dic[columns[f_counter]] = group_df_query_doc.values[0][f_counter].split(':')[1]
            result_doc_dic['rank_num'] =  group_df_query_doc.values[0][0]
            result_dic['docs'].append(result_doc_dic)
            counter +=1
            print('counter', counter)
    result.append(result_dic)

end = time.time()

print('time {}'.format((end - start)/ 60))

with open('./data/processed_fold_{}_threshold_{}.pkl'.format(fold_num, threshold), 'wb') as f:
    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
