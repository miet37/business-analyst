# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 18:33:07 2023

@author: prac
"""
import pandas as pd
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()

import nltk
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords  
import string

from nltk import bigrams
import itertools
import collections
import matplotlib.pyplot as plt
import networkx as nx

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
        
    return topics

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


#########################
# read file
dsf = 'pracuj_analityk_w_kat_translated_1d_dfp_dataset.xlsx'

ds = pd.read_excel(dsf)
ds.info()
ds.columns

# preprocessing

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

# calculate bigrams and network
pos_title_bigrams = bigrams_ds(pos_tit_stat=pos_title_stat)

# plotting network for top_n bigrams
ps_top = 25
pos_title_bigrams_top =  {x: pos_title_bigrams[0][x] for x in list(pos_title_bigrams[0].keys())[:ps_top]}
plot_network(d=pos_title_bigrams_top, font_size=40)



# clustering tabeli sumarycznej
dfg = pos_title_stat['index'] #dataset, unique values
pos_title_topics = mallet_clustering(dfg, output_directory_path = 'm', num_topics = 50)

# word probability within topics
pos_title_word_prob_dict = topic_word_prob_dict(output_directory_path = 'm', num_topics=50, prob_limit = 5)

# preparation for dict for category
pos_title_topics_word_proba_dict = [x[1] for x in pos_title_word_prob_dict] 

##########################
# przypisanie job title do categorii

fcat = 'pracuj_analityk_w_kat_translated_2c_category_from_topics.xlsx'
df_cat1 = pd.read_excel(fcat, sheet_name='Sheet1a')
df_cat1.info()


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

# directory for job category
df_cat = df_cat1[df_cat1['rank']>0]
print(df_cat.info())

cat_dict, cat_dict_flat, cat_keywords_tab = cat_dir(df_cat)

def bigrams_list(x_str):   
    from nltk import bigrams
    
    terms_bigrams = list(bigrams(x_str.split(' ')))
    return terms_bigrams #, bigrams_flat

bigrams_list('environmental social analyst')


# Jaccard index
def jacard_index(dfg):
    
    dfg_cat = pd.DataFrame()
    dfg_cat['doc'] = dfg
    dfg_cat['category'] = 'n'
    dfg_cat['jaccard_idx'] = 0
    dfg_cat['jaccard_calc'] = ''
    
    
    for i, r in dfg_cat.iterrows():
        if r.doc in df_cat['category'].unique():        
            print(i, 'Category = title -->', r.doc )
            dfg_cat.loc[i, 'category' ] = r.doc
    
        else:
            
            intesect_len_max = int(0)
            pos_title_set = set(r['doc'].split(' '))
            print('\n-->',pos_title_set)
            
            cat_wg_jaccard_calc = []
            cat_wg_jaccard_max = 'n'
            for j in cat_keywords_tab.items():       
                
                pos_tit_intersect = pos_title_set.intersection(j[1])
                
                intesect_len = len(pos_tit_intersect)
                if r['doc'].split(' ')[0] in pos_tit_intersect: intesect_len += 1
                cat_wg_jaccard_calc.append(' '.join([j[0], str(intesect_len), ' '.join(pos_tit_intersect)]))
                
                print(j[0], pos_tit_intersect, cat_wg_jaccard[-1])       
                
                if intesect_len > intesect_len_max :
                    print('cat -->',j[0],intesect_len)
                    intesect_len_max = intesect_len
                    cat_wg_jaccard_max = j[0]
                                 
            print('R:', i, 'T:', r['doc'], 'C:',cat_wg_jaccard_max, 'Jidx:', intesect_len_max  )     
            dfg_cat.loc[i, 'category' ] = cat_wg_jaccard_max
            dfg_cat.loc[i, 'jaccard_idx'] = intesect_len_max
            dfg_cat.loc[i, 'jaccard_calc'] = ' '.join(cat_wg_jaccard_calc)
    return dfg_cat        

dfg_cat = jacard_index(dfg)
dfg_cat.to_excel('pracuj_analityk_w_kat_translated_2d_category_pos_title.xlsx') 




'''

       
x = dfg_cat.iloc[0]['pos_title'] 
x in df_cat[df_cat['rank']==0]['category'].to_list()



# Result of mamual work based on 50 topics
analyst_category = {
"business analyst": [0,1,11,14,22,24,32,33,34,38,39,41,42,43,48,49],
"data scientist": [3,6,8,15,18,27,30,40],
"financial analyst": [2,5,7,10,13,16,17,20,23,29,47],
"financial controller": [9,26,28,36,45],
"intern analyst": [35,37],
"security analyst": [12,25,31,44,46],
"system analyst": [4,19,21]
}
print(analyst_category.items())




# calculations
keywords_for_category_dict, dict_for_wc = dict_for_category(analyst_category, pos_title_topics_word_proba_dict)


def word_cloud_fig(dict_for_wc=dict_for_wc):
    #word cloud figure for category
    from wordcloud import WordCloud
    wordcloud = WordCloud(prefer_horizontal=1, random_state=141, width=600, height=300,
                      min_font_size=8,
                      background_color='white').fit_words(dict_for_wc)
    
    plt.figure( figsize=(10,5), facecolor='w')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(cat[0])
    plt.tight_layout(pad=0)
    plt.axis("off")
    plt.show()
    


sect = 1
df_topics = pd.DataFrame(columns=['sector','topics'])
df_topics['topics'] = topics
df_topics['sector'] = sect
df_topics.iloc[0]

# dataset = dfg
df_topic_distributions = pd.DataFrame()
df_topic_distributions['sector'] = sect
df_topic_distributions['doc'] = dfg
df_topic_distributions.iloc[0]

    
print('Topic_distributions',df_topic_distributions.shape)
print('Training_data stat',lmw.print_dataset_stats(dfg))

    
#distribution ... documents belongs to topic
topic_distributions = lmw.load_topic_distributions(path_to_topic_distributions)
print('topic_dist_dics',len(topic_distributions))
print('topics',len(topic_distributions[0]))

#enumarate topics to find max
topic_p_tit = [[(topic_distributions[k][i],i) for i in range(0,len(topics))] for k in range(0,len(topic_distributions))]

#sorted(ax[0], key=lambda s: s[1], reverse=True)[0][0]
topic_prob = [sorted(topic_p_tit[k], key = lambda t: t[0], reverse=True)[0][0] for k in range(len(topic_p_tit))]
topic_no = [sorted(topic_p_tit[k], key = lambda t: t[0], reverse=True)[0][1] for k in range(len(topic_p_tit))]
topic_keys_str = [','.join(df_topics[df_topics['sector']==sect].iloc[x]['topics']) for x in topic_no]

df_topic_distributions['topic_prob'] = topic_prob
df_topic_distributions['topic_no']   = topic_no
df_topic_distributions['topic_keys'] =  topic_keys_str
print(df_topic_distributions.info())
print(df_topic_distributions.iloc[0])

df_topic_distributions['category'] = \
    df_topic_distributions['topic_no'].apply(lambda x: [c for c,v in zip(analyst_category.keys(),
                                                                          analyst_category.values()) \
                                                        if x in v])

df_topic_distributions['category_from_topics'] = \
    df_topic_distributions['topic_no'].apply(lambda x: [v for c,v in zip(analyst_category.keys(),
                                                                          analyst_category.values()) \
                                                        if x in v])

df_topic_distributions.info()
df_topic_distributions.index

df_file = 'pracuj_analityk_w_kat_translated_2a_cat_from_topics.xlsx'
df_topic_distributions.to_excel(df_file+'.xlsx')


# transfering category to oryginal file, ds2 for seftiness
ds2 = ds.merge(df_topic_distributions,
               how='left',
               left_on='pos_tit_en_cleaned',
               right_on='doc')

ds2.info()
ds2.to_excel('pracuj_analityk_w_kat_translated_2b_cat_from_topics_tot.xlsx')

# transfering category to oryginal file
ds = ds.merge(df_topic_distributions[['category','doc']],
               how='left',
               left_on='pos_tit_en_cleaned',
               right_on='doc')

ds.info()

# koniec analizay pos_tit_en_cleaned

# Faza 1 - position level
# Faza 2 - position title
# Faza 3 - Requirments
ds.info()
ds.columns
ds['responsibilities-1_en_cleaned'] = ds['responsibilities-1'].apply(lambda x: pre_proc_lema(x))


#tworzenie tabeli z wartociami uikalnymi
resp_stat = feture_stat(column_str='responsibilities-1_en_cleaned')
print('resp_stat_len', len(resp_stat))

# calculate bigrams
resp_bigrams = bigrams_ds(pos_tit_stat=resp_stat)
plot_network(resp_bigrams, 12)


#######################################




#calculation topics and word's proba

resp_topics = mallet_clustering(dfg = resp_stat['index'], output_directory_path = 'm_resp', num_topics = 10)  
resp_topic_word_proba =  topic_word_prob_dict(output_directory_path = 'm_resp', num_topics=10, prob_limit = 5)


#distribution ... documents belongs to topic
def topic_distrib(dfg = resp_stat['index'],  output_directory_path = 'm_resp', num_topics = 10):
    df_topic_distributions = pd.DataFrame()
    df_topic_distributions['doc'] = dfg
    path_to_topic_distributions     = output_directory_path + '/mallet.topic_distributions.' + str(num_topics)

    topic_distributions = lmw.load_topic_distributions(path_to_topic_distributions)
    
    print('topic_dist_dics',len(topic_distributions))
    print('topics',len(topic_distributions[0]))

    #enumarate topics to find max
    topic_p_tit = [[(topic_distributions[k][i],i) for i in range(0,len(topics))] for k in range(0,len(topic_distributions))]

    #sorted(ax[0], key=lambda s: s[1], reverse=True)[0][0]
    topic_prob = [sorted(topic_p_tit[k], key = lambda t: t[0], reverse=True)[0][0] for k in range(len(topic_p_tit))]
    topic_no = [sorted(topic_p_tit[k], key = lambda t: t[0], reverse=True)[0][1] for k in range(len(topic_p_tit))]
    topic_keys_str = [','.join(df_topics[df_topics['sector']==sect].iloc[x]['topics']) for x in topic_no]

    df_topic_distributions['topic_prob'] = topic_prob
    df_topic_distributions['topic_no']   = topic_no
    df_topic_distributions['topic_keys'] =  topic_keys_str
    #print(df_topic_distributions.info())
    #print(df_topic_distributions.iloc[0])
    
    return df_topic_distributions

resp_topic_distrib = topic_distrib(dfg = resp_stat['index'],  output_directory_path = 'm_resp', num_topics = 10)
resp_topic_distrib.to_excel('xx.xlsx')

i=0
for t in resp_topics:
    print(i,t)
    for cat in 
    set(t).intersection()
    i+=1





from sentence_similarity import sentence_similarity
sentence_a = "paris is a beautiful city"
sentence_b = "paris is a grogeous city"
model=sentence_similarity(model_name='distilbert-base-uncased',embedding_type='cls_token_embedding')
score=model.get_score(sentence_a,sentence_b,metric="cosine")
print(score)

s1 = ' '.join(['analyst','security','kyc','client','aml','prevention','anti','revenue','know','soc','designer','fraud'])
s2 = ' '.join(['network', 'administrator', 'performance', 'sap', 'analyst',  'tester',  'computer', 'center','key', 'user'])
score=model.get_score(s1,s2,metric="cosine")
print(score)

    df_topic_distributions['category'] = \
       df_topic_distributions['topic_no'].apply(lambda x: [c for c,v in zip(analyst_category.keys(),
                                                                          analyst_category.values()) \
                                                        if x in v])

df_topic_distributions['category_from_topics'] = \
    df_topic_distributions['topic_no'].apply(lambda x: [v for c,v in zip(analyst_category.keys(),
                                                                          analyst_category.values()) \
                                                        if x in v])

df_topic_distributions.info()
df_topic_distributions.index

df_file = 'pracuj_analityk_w_kat_translated_2a_cat_from_topics.xlsx'
df_topic_distributions.to_excel(df_file+'.xlsx')


# transfering category to oryginal file, ds2 for seftiness
ds2 = ds.merge(df_topic_distributions,
               how='left',
               left_on='pos_tit_en_cleaned',
               right_on='doc')

ds2.info()
ds2.to_excel('pracuj_analityk_w_kat_translated_2b_cat_from_topics_tot.xlsx')

# transfering category to oryginal file
ds = ds.merge(df_topic_distributions[['category','doc']],
               how='left',
               left_on='pos_tit_en_cleaned',
               right_on='doc')

ds.info()

# koniec analizay pos_tit_en_cleaned















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





















ds[['responsibilities-1','responsibilities-1_en_cleaned']].to_excel('xx.xlsx') 




xf1 = 'pracuj_analityk_w_kat_translated_1.xlsx'
# pos_title translated to pos_title_en, 482 records, category
xdf1 = pd.read_excel(xf1, sheet_name='Sheet1')
xdf1.info()

xf2 = 'pracuj_analityk_w_kat_translated_2.xlsx'
# all translated, there is no category
xdf2 = pd.read_excel(xf2, sheet_name='Sheet1')
xdf2.columns.to_list()

xy_cat = xdf1[['pos_href', 'pos_tit_en', 'category' ]]
# selection pos_href as key, pos_tit_en, category - for training
xy_cat.info()

xdf2 = xdf2.merge(xy_cat, how='left', on='pos_href')
xdf2.info()

# filtrowanie rekordów be offer_tab
xdf2 = xdf2[xdf2.offer_tab!="[[''], 0]"].reset_index()
xdf2.to_excel('pracuj_analityk_w_kat_translated_3.xlsx')


xdf2 = pd.read_excel('pracuj_analityk_w_kat_translated_3.xlsx')

# preprocessing by pre_proc_lema(x)
xdf2['pos_tit_en_x_cleaned'] = xdf2['pos_tit_en_x'].apply(lambda x: pre_proc_lema(x))
xdf2[['pos_tit_en_x_cleaned','pos_tit_en_x','category']].to_excel('pracuj_analityk_w_kat_translated_3_preproc.xlsx')

# ----> strat classyfication
xdf2 = pd.read_excel('pracuj_analityk_w_kat_translated_3.xlsx')

# Preprocessing kolumny pos_tit_en 

Xtxt = xdf2.pos_tit_en_x_cleaned
y = xdf2.category 
print(y.unique())
print('Is None  in X', Xtxt.isna().sum(),'y', y.isna().sum())


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(Xtxt).toarray()

# data secection for training, eliminating records where y=None
xx = y.notna()

Xt = [a for a,b in zip(X,xx) if b]
yt = [a for a,b in zip(y,xx) if b]
print('Len','X',len(Xt),'y',len(Xt))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xt, yt, test_size=0.25, random_state=0, stratify=yt)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

# MultinomialNB
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

# SGDClassifier
from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier()
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                           max_iter=5, tol=None)),
     ])

text_clf.fit(twenty_train.data, twenty_train.target)
Pipeline(...)
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)


from sklearn.model_selection import GridSearchCV
parameters = {
     'vect__ngram_range': [(1, 1), (1, 2)],
     'tfidf__use_idf': (True, False),
     'clf__alpha': (1e-2, 1e-3),
     }
gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)


y_pred_all = classifier.predict(X)

xdf2['category_ml'] = y_pred_all
xdf2['documents_ml'] = Xtxt
xdf2.to_excel('pracuj_analityk_w_kat_translated_3_category_ml.xlsx')

# final file for cattegory
fx = "pracuj_analityk_w_kat_translated_3_category_ml.xlsx"
dfx = pd.read_excel(fx)

print(dfx.shape)
print(dfx.info())
dfx.columns.to_list()


# combining additional-modules-3i together
dfx['additional-module-3x_en'] = ''
for i in range(3,10):
    dfx['additional-module-3x_en'] = dfx['additional-module-3x_en'] + ' '+ \
    + dfx['additional-module-'+str(i)+'_en'].fillna('')
    print(i,'additional-module-'+str(i),
          dfx[dfx['additional-module-3x_en'].str.strip()!='']['additional-module-3x_en'].count())

# combining offered and benefits together
print('offered and benefits before',dfx['offered-1_en'].notna().sum(), dfx['benefits-1_en'].notna().sum(), )

dfx['offered-1_en'] = dfx['offered-1_en'].fillna('') +' '+ dfx['benefits-1_en'].fillna('')
print('offered after',dfx[dfx['offered-1_en']!=' ']['offered-1_en'].count() )

# replace pos_title by preprocessed 

dfx['pos_tit_en'] = dfx['documents_ml']

for field in [
    'pos_add_info_en',
    'offer_tab_cat',
    'technologies-1_en',
    'responsibilities-1_en',
    'requirements-1_en',
    'development-practices-1_en',
    'additional-module-1_en',
    'additional-module-2_en',
    'additional-module-3x_en',
    'offered-1_en',
    'training-space-1_en',
    ]:
    print(field)
    x = dfx[field]
    dfx[field] = pre_proc_lema(x)


Out_file =  [
    'pos_href',
    'pos_tit_en'
    'pos_add_info_en',
    'offer_tab_cat',
    'technologies-1_en',
    'responsibilities-1_en',
    'requirements-1_en',
    'development-practices-1_en',
    'additional-module-1_en',
    'additional-module-2_en',
    'additional-module-3x_en',
    'offered-1_en',
    'training-space-1_en',
    'pos_tit_en_y',
    'category',
    'category_ml',
    'documents_ml'
    ]

dfx[Out_file].fillna('').to_excel('pracuj_for_tm.xlsx')

