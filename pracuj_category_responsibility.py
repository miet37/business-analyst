# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 18:33:07 2023

@author: prac
"""
import pandas as pd
#import numpy as np
import re
from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()

import nltk
nltk.download('stopwords')
#import pickle
from nltk.corpus import stopwords  
import string

from nltk import bigrams
#import itertools
#import collections
#import matplotlib.pyplot as plt
#import networkx as nx

from sentence_similarity import sentence_similarity
model=sentence_similarity(model_name='distilbert-base-uncased',embedding_type='cls_token_embedding')
# https://pypi.org/project/sentence-similarity/

STOPWORDS = set(stopwords.words('english') + list(string.punctuation))
STOPWORDS.remove('it')
   
def pre_proc_lema(x):
      
    
    document = re.sub(r'\W', ' ', x)  # Remove all the special characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document) # remove all single characters  
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) # Remove single characters from the start
    document = re.sub(r'\s+', ' ', document, flags=re.I) # Substituting multiple spaces with single space
    
    # Removing prefixed 'b'
    # document = re.sub(r'^b\s+', '', document)   
    
    document = document.lower() # Converting to Lowercase
    document = document.replace('analityk','analyst')
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document if (len(word)>1) & (word not in STOPWORDS)]    
    
    doc_final = []
    for i in document:
        if i not in doc_final:
            doc_final.append(i) 
        
    document = ' '.join(doc_final)

    return document

#tworzenie tabeli z wartociami uikalnymi
def feture_stat(column_str='pos_tit_en_cleaned'): 
    pos_tit_stat = ds[column_str].value_counts().reset_index()
    print('Feature stat len',column_str,len(pos_tit_stat))
    
    #pos_tit_stat['len'] = pos_tit_stat['index'].apply(lambda x: len(str(x).split(' ')))
    #pos_tit_stat['len'].plot(kind='hist', grid=True, bins=20, rwidth=1 )
    #plt.title(column_str)
    #plt.xlabel('Words Counts')
    #plt.ylabel('Frequency')
    #plt.grid(axis='y', alpha=0.75)
    # maxfreq = pos_tit_stat.max()
    # Set a clean upper y-axis limit.
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)    
    
    return pos_tit_stat

# bigrams
# Create list of lists containing bigrams
def bigrams_ds(pos_tit_stat): 
    # pos_tit_stat=resp_stat    
    from nltk import bigrams
    import itertools
    import collections
    
    terms_bigram = [list(bigrams(x.split(' '))) for x in pos_tit_stat['index'][0:len(pos_tit_stat)]]
    
    # Flatten list of bigrams in clean tweets
    bigrams_flat = list(itertools.chain(*terms_bigram))
    
    # only collocation with analyst - unrem if you want to 
    # bigrams_flat = [x for x in bigrams_flat if 'analyst' in x]
    
    cust_stops = ['owner','product', 'real','estate'] 
    bigram_stops = [x for x in bigrams_flat if ((x[0] in cust_stops) or (x[1] in cust_stops)) ]
    bigrams_flat = [x for x in bigrams_flat if x not in bigram_stops]
    
    # Create counter of words in clean bigrams
    bigram_counts = collections.Counter( bigrams_flat )
    
    
    #_analyst
    # bigram_counts = {key:value for (key,value) in bigram_counts.items() if 'analyst' in key}
    
    print(bigram_counts.most_common(20))
    
    bigram_df = pd.DataFrame(bigram_counts.most_common(150), columns=['bigram_terms', 'count'])
    #print(bigram_df)
    
    # Create dictionary of bigrams and their counts
    d = bigram_df.set_index('bigram_terms').T.to_dict('records')
    
    return d

# Create network plot 
def plot_network(d, font_size=10):

    import matplotlib.pyplot as plt
    import networkx as nx
    G = nx.Graph()

    # Create connections between nodes
    for k, v in d.items():
        G.add_edge(k[0], k[1], weight=(v * 20))

        #G.add_node("china", weight=100)

    fig, ax = plt.subplots(figsize=(40, 32))
    pos = nx.spring_layout(G, k=1.2)

    # Plot networks
    nx.draw_networkx(G, pos,
                 font_size=font_size,
                 width=1,
                 edge_color='grey',
                 node_color='grey',    #'purple',
                 with_labels = True,
                 node_size = 10,
                 ax=ax)

    # Create offset labels
    #for key, value in pos.items():
    #    x, y = value[0]-.0135, value[1]+.045
    #    ax.text(x, y,
    #        s=key,
    #        bbox=dict(facecolor='white', alpha=0),
    #        horizontalalignment='left', fontsize=10)
    
   
    plt.savefig("xxx.png", format="png", dpi=300)
    plt.show()
    
    return True

# clustering
def mallet_clustering(dfg, output_directory_path = 'm', num_topics = 50):   
    
    import os
    import little_mallet_wrapper as lmw
    path_to_mallet = 'C:/Users/prac/Documents/Programy/mallet-2.0.8/bin/mallet.bat' #.bat is important in windows
       
    try: 
        os.mkdir(f"{output_directory_path}")
    except OSError as error:
        print(error)
    
    path_to_training_data           = output_directory_path + '/training.txt'
    path_to_formatted_training_data = output_directory_path + '/mallet.training'
    path_to_model                   = output_directory_path + '/mallet.model.' + str(num_topics)
    path_to_topic_keys              = output_directory_path + '/mallet.topic_keys.' + str(num_topics)
    path_to_topic_distributions     = output_directory_path + '/mallet.topic_distributions.' + str(num_topics)
    path_to_word_weights            = output_directory_path + '/mallet.word_weights.' + str(num_topics)
    path_to_diagnostics             = output_directory_path + '/mallet.diagnostics.' + str(num_topics) + '.xml'   
    
    print('Dataset stat',lmw.print_dataset_stats( dfg ))
    print('Commputing ... ', len( dfg))
     
    lmw.import_data(path_to_mallet, path_to_training_data, path_to_formatted_training_data, dfg)
    
    lmw.train_topic_model(path_to_mallet,
                       path_to_formatted_training_data,
                       path_to_model,
                       path_to_topic_keys,
                       path_to_topic_distributions,
                       path_to_word_weights,
                       path_to_diagnostics,
                       num_topics)
     
    #Display Topics and Top Words
    topics = lmw.load_topic_keys(path_to_topic_keys)
    #for i,t in zip(range(len(topics)),topics):
    #    print('-->topic '+str(i),' '.join(t))
    # topic distribution to docunemts 
    
    df_topics = pd.DataFrame(columns=['sector','topics'])
    df_topics['topics'] = topics
    df_topics['sector'] = '1'
    
    # dataset = dfg
    df_topic_distribution = pd.DataFrame()
    df_topic_distribution['doc'] = dfg
        
    print('Topic_distributions shape:',df_topic_distribution.shape)
        
    #distribution ... documents belongs to topic
    topic_distribution = lmw.load_topic_distributions(path_to_topic_distributions)
    print('topic_dist_dics', len(topic_distribution),'topics', len(topic_distribution[0]))
    
    #enumarate topics to find max
    topic_p_tit = [[(topic_distribution[k][i],i) for i in range(0,len(topics))] for k in range(0,len(topic_distribution))]
    
    #sorted(ax[0], key=lambda s: s[1], reverse=True)[0][0]
    topic_prob = [sorted(topic_p_tit[k], key = lambda t: t[0], reverse=True)[0][0] for k in range(len(topic_p_tit))]
    topic_no = [sorted(topic_p_tit[k], key = lambda t: t[0], reverse=True)[0][1] for k in range(len(topic_p_tit))]
    topic_keys_str = [','.join(df_topics[df_topics['sector']=='1'].iloc[x]['topics']) for x in topic_no]
    
    df_topic_distribution['topic_prob'] = topic_prob
    df_topic_distribution['topic_no']   = topic_no
    df_topic_distribution['topic_keys'] =  topic_keys_str
                
    return topics, df_topic_distribution

# word probability within topics
def topic_word_prob_dict(output_directory_path = 'm_resp', num_topics=50, prob_limit = 5):
    
    import little_mallet_wrapper as lmw
    
    topic_word_probability_dict = lmw.load_topic_word_distributions(output_directory_path + '/mallet.word_weights.' + str(num_topics))
    print('\ntopic_word_probability_dict len', len(topic_word_probability_dict))

    topic_word_prop = []
    print('** _word_probability_dict *********')
    for _topic, _word_probability_dict in topic_word_probability_dict.items():
        print('\nTopic', _topic, '* ',end='')
        #print('\n',end='')
        word_prob_dict = {}
        word_prob_sum = 0
        for _word, _probability in sorted(_word_probability_dict.items(), key=lambda x: x[1], reverse=True)[:10]:
            word_prob_dict[_word] = _probability
            if round(_probability*1000, 0) >= prob_limit:
                word_prob_sum += _probability
                print(_word, '(', round(_probability*1000, 0), ') ',end='') 
                #print(_word, ',',end='') 
        topic_word_prop.append([_topic, word_prob_dict])    
    return topic_word_prop

# build 
#
def dict_for_category(analyst_category, topic_word_prop):
    keywords_for_category = []
    keywords_for_category_words = []
    for cat in analyst_category.items():
        print(cat[0])
        
        dict_for_wc = {}
        ind = 0
        for t in topic_word_prop:
            #print(ind)
            if ind in cat[1]:
                for i in [(x[0],x[1]) for x in t.items()][0:4]:
                    if i[1] >= 0.05:
                        if i[0] in dict_for_wc:
                            dict_for_wc[i[0]] += i[1]
                        else:
                            dict_for_wc[i[0]] = i[1]
            ind += 1
        
        keywords_for_category.append([cat[0], sorted(dict_for_wc.items(), key=lambda x: x[1], reverse=True)])
        keywords_for_category_words.append([cat[0], [x[0] for x in sorted(dict_for_wc.items(), key=lambda x: x[1], reverse=True)]])
        
        #eliminate duplicates
        #locking for unique keywords
        keywords_for_category_dict = {}
        keywords_for_category_del = []
        
        for k in keywords_for_category_words:
            print(k)
            for w in k[1]:            
                if w in keywords_for_category_dict:
                    keywords_for_category_del.append([w, k[0]])
                    #del_key = keywords_for_category_dict.pop(w,None)
                    print('duplicated key')
                    keywords_for_category_dict[w] += [k[0]]
                else:
                    keywords_for_category_dict[w] = [k[0]]
                    print('dict in',w,k[0])

    
    #print( keywords_for_category_dict.keys() )
    return keywords_for_category_dict, dict_for_wc


def cat_dir(df_cat):
    # {keyword: [categorirs]}
    cat_dict = {}
    cat_dict_flat = {}
    for i,r in df_cat.iterrows():
        if r.keyword not in cat_dict:
            cat_dict[r.keyword] = [{r.category: r.weight}]
            cat_dict_flat[r.keyword] = [r.category]
        else:
            cat_dict[r.keyword] += [{r.category: r.weight}]
            cat_dict_flat[r.keyword] += [r.category]
    
    # {category: [keywords]}
    cat_list = df_cat['category'].unique()
    cat_keywords_tab = {}
    for cat in cat_list:
        if cat not in cat_keywords_tab:
            cat_keywords_tab[cat] = df_cat[df_cat['category']==cat]['keyword'].to_list()
        else:
            cat_keywords_tab[cat] += df_cat[df_cat['category']==cat]['keyword'].to_list()
    
    return cat_dict, cat_dict_flat, cat_keywords_tab



def bigrams_list(x_str):   
#    from nltk import bigrams
    
    terms_bigrams = list(bigrams(x_str.split(' ')))
    return terms_bigrams #, bigrams_flat

# test bigrams_list('environmental social analyst')


# Jaccard index
def jacard_index(dfg, df_cat, cat_keywords_tab, lsuffix='',lcos_score=False):
    
    
    dfg_cat = pd.DataFrame()
    dfg_cat['doc'] = dfg
    dfg_cat[lsuffix+'category'] = 'n'
    dfg_cat[lsuffix+'jaccard_idx'] = 0
    dfg_cat[lsuffix+'jaccard_calc'] = ''
    dfg_cat[lsuffix+'cos_score_calc'] = ''
    dfg_cat[lsuffix+'cos_idx'] = 0
    dfg_cat[lsuffix+'cos_category'] = 'n'
    
    
    for i, r in dfg_cat.iterrows():
        if r.doc in df_cat['category'].unique():        
            print(i, 'Category = title ->', r.doc )
            dfg_cat.loc[i, lsuffix+'category' ] = r.doc
    
        else:
            
            
            pos_title_set = set(r['doc'].split(' '))
            print('\n------>', pos_title_set )
            
            cat_cos_score_calc = []
            cat_cos_score_max = int(0)
            cat_wg_cos_max = 'n'
            
            cat_wg_jaccard_calc = []
            intesect_len_max = int(0)
            cat_wg_jaccard_max = 'n'
            
            for j in cat_keywords_tab.items():       
                
                pos_tit_intersect = pos_title_set.intersection(j[1])
                
                intesect_len = len(pos_tit_intersect)
                if r['doc'].split(' ')[0] in pos_tit_intersect: intesect_len += 1
                
                cat_wg_jaccard_calc.append(' c:'+' '.join([j[0], ' ji:'+str(intesect_len), ' Int:'+' '.join(pos_tit_intersect)])) 
                
                cos_score = model.get_score(r['doc'], ' '.join(j[1]), metric="cosine") if lcos_score else 0
                cat_cos_score_calc.append('cos:'+' '.join([j[0], ' cos:'+str( cos_score )]))
                
                print(' kw:', j[0], pos_tit_intersect)       
                
                if intesect_len > intesect_len_max :
                    print('cat -->',j[0],intesect_len)
                    intesect_len_max = intesect_len
                    cat_wg_jaccard_max = j[0]
                
                if cos_score > cat_cos_score_max:
                    cat_cos_score_max = cos_score
                    cat_wg_cos_max = j[0]
                                 
            print('R:', i, 'T:', r['doc'], 'Cat_jacc:',cat_wg_jaccard_max, 'Jidx:', intesect_len_max  )
            print('Cos:', 'C:'+' '.join(cat_cos_score_calc),'\nCat_cos_max:', cat_wg_cos_max  )
            
            dfg_cat.loc[i, lsuffix+'category' ] = cat_wg_jaccard_max
            dfg_cat.loc[i, lsuffix+'jaccard_idx'] = intesect_len_max
            dfg_cat.loc[i, lsuffix+'jaccard_calc'] = ' '.join(cat_wg_jaccard_calc)
            dfg_cat.loc[i, lsuffix+'cos_score_calc'] = ' '.join(cat_cos_score_calc)
            dfg_cat.loc[i, lsuffix+'cos_idx'] = cat_cos_score_max
            dfg_cat.loc[i, lsuffix+'cos_category'] = cat_wg_cos_max
            
            pos_tit_intersect = pos_title_set.difference(cat_keywords_tab[cat_wg_jaccard_max]) if cat_wg_jaccard_max!='n' else ['n']
            
            dfg_cat.loc[i, lsuffix+'jaccard_diff'] = ' '.join(pos_tit_intersect)
                      
            
    return dfg_cat      


# from sentence_similarity import sentence_similarity
# sentence_a = "paris is a beautiful city"
# sentence_b = "paris is a grogeous city"
# model=sentence_similarity(model_name='distilbert-base-uncased',embedding_type='cls_token_embedding')
# score=model.get_score(sentence_a,sentence_b,metric="cosine")
# print(score)



# ===================================
# directory for job category ======
# ---> Category Fixed based on pos_title --> do użycia dalej
# przypisanie job title do categorii

fcat = 'pracuj_analityk_w_kat_translated_2c_category_from_topics.xlsx'
df_cat1 = pd.read_excel(fcat, sheet_name='Sheet1a')
df_cat1.info()

df_cat = df_cat1[df_cat1['rank']>0]
print(df_cat.info())
print(df_cat.groupby('category').count())
# ================

cat_dict, cat_dict_flat, cat_keywords_tab = cat_dir(df_cat)


#########################
# read file
dsf = 'pracuj_analityk_w_kat_translated_1d_dfp_dataset.xlsx'

ds = pd.read_excel(dsf)
ds.info()
ds.columns

# >>>======= analiza pos_tit_en
# preprocessing pos_tit_en

ds['pos_tit_en_cleaned'] = ds['pos_tit_en'].apply(lambda x: pre_proc_lema(x))
ds['pos_tit_en_cleaned'] = ds['pos_tit_en_cleaned'].str.replace('junior','').str.strip()
ds['pos_tit_en_cleaned'] = ds['pos_tit_en_cleaned'].str.replace('senior','').str.strip()
ds['pos_tit_en_cleaned'] = ds['pos_tit_en_cleaned'].str.replace('german','').str.strip()
ds['pos_tit_en_cleaned'] = ds['pos_tit_en_cleaned'].str.replace('english','').str.strip()
ds['pos_tit_en_cleaned'] = ds['pos_tit_en_cleaned'].str.replace('spanish','').str.strip()
ds['pos_tit_en_cleaned'] = ds['pos_tit_en_cleaned'].str.replace('french','').str.strip()
ds['pos_tit_en_cleaned'] = ds['pos_tit_en_cleaned'].str.replace('emea','').str.strip()
ds['pos_tit_en_cleaned'] = ds['pos_tit_en_cleaned'].str.replace('department','').str.strip()
ds['pos_tit_en_cleaned'] = ds['pos_tit_en_cleaned'].str.replace('global','').str.strip()

ds.to_excel('pracuj_analityk_w_kat_translated_1e_dfp_dataset.xlsx')


# tworzenie tabeli z wartociami uikalnymi and plotting histogram
pos_title_stat = feture_stat(column_str='pos_tit_en_cleaned')
pos_title_stat['len'] = pos_title_stat['index'].apply(lambda x: len(str(x).split(' ')))

# ploting histogram
pos_title_stat['len'].plot(kind='hist', grid=True, bins=10)
pos_title_stat.info()
pos_title_stat.to_excel('pracuj_analityk_w_kat_stat_pos_title_stat.xlsx')

# calculate bigrams and network
pos_title_bigrams = bigrams_ds(pos_tit_stat=pos_title_stat)

# plotting network for top_n bigrams
ps_top = 25
pos_title_bigrams_top =  {x: pos_title_bigrams[0][x] for x in list(pos_title_bigrams[0].keys())[:ps_top]}
plot_network(d=pos_title_bigrams_top, font_size=40)



# clustering tabeli sumarycznej
pos_title_dfg = pos_title_stat['index'] #dataset, unique values
pos_title_topics = mallet_clustering(pos_title_dfg, output_directory_path = 'm', num_topics = 50)

# word probability within topics
pos_title_word_prob_dict = topic_word_prob_dict(output_directory_path = 'm', num_topics=50, prob_limit = 5)

# preparation for dict for category
pos_title_topics_word_proba_dict = [x[1] for x in pos_title_word_prob_dict] 

# Jaccard and cos index
pos_title_dfg_cat = jacard_index(pos_title_dfg, df_cat, cat_keywords_tab, lsuffix='pos_tit_')

# adding to dataset
ds = ds.join(pos_title_dfg_cat.set_index('doc'), on='pos_tit_en_cleaned')
ds.info()

pos_title_dfg_cat.to_excel('pracuj_analityk_w_kat_translated_2d_category_pos_title2.xlsx') 

ds.to_excel('pracuj_analityk_w_kat_translated_2d_category_pos_title2_ds.xlsx') 




# >>>======= analiza responsibilities-1

ds['resp_cleaned'] = ds['responsibilities-1'].apply(lambda x: pre_proc_lema(x))

# tworzenie tabeli z wartociami uikalnymi and plotting histogram
resp_stat = feture_stat(column_str='resp_cleaned')
resp_stat['resp_len'] = resp_stat['index'].apply(lambda x: len(str(x).split(' ')))
resp_stat.info()
resp_stat.to_excel('pracuj_analityk_w_kat_stat_resp_stat.xlsx')


# ploting histogram
resp_stat['resp_len'].plot(kind='hist', grid=True, bins=10)

# calculate bigrams and network
resp_bigrams = bigrams_ds(resp_stat)

# plotting network for top_n bigrams
ps_top = 25
resp_bigrams_top =  {x: resp_bigrams[0][x] for x in list(resp_bigrams[0].keys())[:ps_top]}
plot_network(d=resp_bigrams_top, font_size=40)



# clustering tabeli sumarycznej
resp_dfg = resp_stat['index'] #dataset, unique values
resp_topics, resp_topics_distr = mallet_clustering(resp_dfg, output_directory_path = 'm_resp', num_topics = 50)
# resp_topics_distr.to_excel('xx.xlsx')

# word probability in topics
resp_word_prob_dict = topic_word_prob_dict(output_directory_path = 'm_resp', num_topics=50, prob_limit = 5)

# topics for categorization
resp_topics_str = [' '.join(x[1].keys()) for x in resp_word_prob_dict] 
resp_topics_df = pd.DataFrame(resp_topics_str, columns=['resp'])



#categoryzacja 50 topiców na podstawie 10 tokenów
# jacard_index(pos_title_dfg, df_cat, cat_keywords_tab, lsuffix='pos_tit_', lcos_score=False)

resp_dfg_cat = jacard_index(resp_topics_str, df_cat, cat_keywords_tab, lsuffix='resp_')
resp_dfg_cat.to_excel('pracuj_analityk_w_kat_translated_2e_category_resp_topics.xlsx') 


#categoryzacja doc na podstawie topiców 20 tokenów
print(resp_topics_distr.columns)
reps_t20 = resp_topics_distr['topic_keys'].apply(lambda x: x.replace(',',' '))

resp_dfg_cat20 = jacard_index(reps_t20, df_cat, cat_keywords_tab, lsuffix='resp_')
resp_dfg_cat20.info()
resp_dfg_cat20.to_excel('pracuj_analityk_w_kat_translated_2e_category_resp_topics20.xlsx') 

#cat_keywords_tab['business analyst']

resp_dfg_kw_diff = resp_dfg_cat20[['resp_category','resp_jaccard_diff']].drop_duplicates()

#wyszukiwanie keywordsów powtarzalnych
diff_kw_dt = {}
for c in resp_dfg_kw_diff['resp_category'].unique():
    diff_kw_d = {}
    for d in resp_dfg_kw_diff[resp_dfg_kw_diff['resp_category']==c]['resp_jaccard_diff']:
        for t in d.split():
            if t not in diff_kw_d:
                diff_kw_d[t] = 1
            else:
                diff_kw_d[t] += 1
    if c not in diff_kw_dt:
        diff_kw_dt[c] = sorted(diff_kw_d.items(), key=lambda x: x[1], reverse=True) 
        
    else:
        diff_kw_dt[c] += diff_kw_d

#usuwanie wspólnych      
comm_kw = []
for i in diff_kw_dt.keys():
    for j in diff_kw_dt.keys():
        if i!=j:
            ckw = set([x[0] for x in diff_kw_dt[i]]).intersection([x[0] for x in diff_kw_dt[j]])
            print(i,j,ckw) 
            comm_kw += list(ckw)
       
print(len(comm_kw))        
print(len(set(comm_kw)))               
        
comm_kw_uq = set(comm_kw)       
print(len(comm_kw_uq))    
       
# usuwanie wspólnych z diff_kw_dt
diff_kw_dtt = {}
for i in  diff_kw_dt.keys():
    ddif = set([x[0] for x in diff_kw_dt[i]]).difference(comm_kw_uq)
    print('-->',diff_kw_dt[i], '>', ddif)
    diff_kw_dtt[i] = list(ddif)

max_dl = max([len(x[1]) for x in diff_kw_dtt.items()] )
diff_kw_dtt2 = {x[0]: x[1]+['' for x in range(max_dl-len(x[1]))] for x in diff_kw_dtt.items() }
        
diff_kw_df = pd.DataFrame( diff_kw_dtt2 ) 
diff_kw_df.to_excel('pracuj_analityk_w_kat_translated_2f_category_resp_add_kw.xlsx')    
        
       
# ----> jaccard dfg
# for resp_dfg

# Jaccard and cos index
resp_dfg_cat = jacard_index(resp_dfg, df_cat, cat_keywords_tab, lsuffix='resp_')

# adding to dataset
ds = ds.join(resp_dfg_cat.set_index('doc'), on='resp_cleaned')
ds.info()

resp_dfg_cat.to_excel('pracuj_analityk_w_kat_translated_2f_category_pos_title2.xlsx') 
ds.to_excel('pracuj_analityk_w_kat_translated_2f_category_pos_title2_ds.xlsx') 


























