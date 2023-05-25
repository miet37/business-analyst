# -*- coding: utf-8 -*-
"""
Translations
mp.27.03.2023
"""

import numpy as np
import pandas as pd


from deep_translator import GoogleTranslator
print('test translation',GoogleTranslator(source='auto', target='en').translate("kot i pies") )


txt = '• Przetwarzanie zagadnień matematyki aktuarialnej i finansowej'
txt += '• Sporządzanie raportów aktuarialnych dla zakładowych świadczeń socjalnych w Niemczech i w Polsce'
txt += '• Bieżąca obsługa firmowych programów emerytalnych naszych klientów' 
txt += '• Jeśli masz odpowiednie kwalifikacje i jesteś zainteresowany: Udział w naszych projektach oprogramowania'
#print(GoogleTranslator(source='auto', target='en').translate(txt) )

txt = "[['https://www.pracuj.pl/praca/actuarial-consultant-warszawa,oferta,1002459469'], 1, ['responsibilities-1', ['• Przetwarzanie zagadnień matematyki aktuarialnej i finansowej', '• Sporządzanie raportów aktuarialnych dla zakładowych świadczeń socjalnych w Niemczech i w Polsce', '• Bieżąca obsługa firmowych programów emerytalnych naszych klientów', '• Jeśli masz odpowiednie kwalifikacje i jesteś zainteresowany: Udział w naszych projektach oprogramowania']], ['requirements-1', ['• Znajomość języka niemieckiego na dobrym poziomie (co najmniej B1)', '• Pomyślnie ukończone studia matematyczne (licencjackie/magisterskie) lub studia naukowe o profilu matematycznym', '• Wysoka motywacja i chęć uczenia się', '• Wysoki poziom umiejętności komunikacyjnych i orientacji na klienta']], ['offered-1', [' • Zatrudnienie na podstawie umowy o pracę oraz atrakcyjne świadczenia socjalne', ' • Interesujące i interdyscyplinarne projekty w firmie doradczej zarządzanej przez właściciela', ' • Wspierające środowisko pracy oparte na wzajemnym uznaniu i współpracy w wysoce zmotywowanym i doskonale wykwalifikowanym zespole', ' • Elastyczna koncepcja miejsca pracy i czasu pracy dostosowana do Twoich potrzeb', ' • Kompleksowe wprowadzenie i wdrożenie przez doświadczonych kolegów', ' • Wynagrodzenie zależne od umiejętności i zaangażowania', ' • Zróżnicowane możliwości szkolenia dostosowane do Twoich zainteresowań i wcześniejszej wiedzy', '', 'Wzbudziliśmy Twoje zainteresowanie?', '', 'W takim razie czekamy na Twoją aplikację. Wyślij swoje CV w języku niemieckim, angielskim lub polskim.', '', 'Prosimy o przesyłanie dokumentów aplikacyjnych za pomocą przycisku aplikowania.']], ['about-us-1', ['Neuburger & Partner (www.neuburger.com) jest renomowaną firmą doradczą w zakresie zakładowych programów emerytalnych z biurami w Monachium, Norymberdze, a obecnie także w Warszawie (jako Neuburger & Partners Actuarial Services Sp. z o.o.).', '', 'Nasza grupa spółek oferuje usługi w całym spektrum matematyki aktuarialnej i finansowej oraz związane z tym doradztwo informatyczne. Przedmiotem naszej pracy jest ocena świadczeń socjalnych firm naszych klientów zgodnie z krajowymi i międzynarodowymi przepisami o rachunkowości, a także szeroki zakres usług w ramach obsługi firmowych programów emerytalnych naszych klientów.', '', 'Wraz z naszą spółką zależną VSE (Versicherungsmathematische Softwareentwicklung GmbH) oferujemy również rozwiązania programowe do oceny zobowiązań emerytalnych.', '', 'Nasz zespół w Warszawie będzie początkowo wspierał nasze działania na rynku niemieckim, ale w przyszłości będzie także rozwijał nowe obszary biznesowe w Polsce.', '', 'Do tego potrzebujemy aktywnego wsparcia ze strony dobrze wykwalifikowanych pracowników ze znajomością języka niemieckiego.']]]"
#print(GoogleTranslator(source='auto', target='en').translate(txt) )



def pp(text_col):
    return text_col.apply(lambda x: ' '.join([w.lower() for w in x.split()]))


#Datset

#rf_pop = "C:/MP/Pracuj/pracuj_analityk_w_kat_34.xlsx_sample1_f.xlsx"
#rf = "C:/MP/Pracuj/pracuj_analityk_w_kat_47.xlsx_sample2_f.xlsx"

rf = "C:/MP/Pracuj/pracuj_analityk_w_kat_sample_cum.xlsx" #dane połączone w arkuszu sheet1


dfx = pd.read_excel(rf, sheet_name="Sheet1")
dfx.columns
dfx.shape

dfp = dfx[['pos_title', 'pos_href', 'pos_add_info','offer_tab']].copy(deep=True)
dfp.columns

#preprocessing
dfp.dropna()
dfp.shape
dfp.columns
dfp.info()

#translations
def translate_list(dfp_list, verbose=False):
    tit_en = []
    il = len(dfp_list)
    i=0
    for tit in dfp_list:
        if pd.notna(tit):
            tit_en_i = GoogleTranslator(source='auto', target='en').translate(tit[:4990]) 
        else:
            tit_en_i = tit
        tit_en.append(tit_en_i)
        print(i,'/',il)
      #  with open('translate_temp'+dfp_list.name+'.txt','a',encoding='utf-8') as ft:
      #      ft.write('\n'+tit_en_i)
        if verbose: print(i,'of',il,'s:',tit, 't:',tit_en[-1])
        i+=1
    return tit_en

dfp['pos_add_info'] = dfp['pos_add_info'].apply(lambda x: eval(x)[0][0].strip())

# ---> translations dfp.pos_add_info
trans_tab = translate_list(dfp.pos_add_info, verbose=True)
dfp['pos_add_info_en'] = trans_tab

# ---> translations dfp.pos_title
trans_tab = translate_list(dfp.pos_title, verbose=True)
dfp['pos_tit_en'] = trans_tab

dfp.to_excel('pracuj_analityk_w_kat_translated_1.xlsx')


# Part 2
dfp = pd.read_excel('pracuj_analityk_w_kat_translated_1.xlsx',sheet_name='Sheet1')
print(dfp.info())


#str(my_byte_str, 'utf-8')
#stxt = dfp['offer_tab']
#dfp['offer_tab'] =  [str(s.encode('ascii', 'ignore'),'utf-8') for s in stxt.str.decode('unicode_escape')]

# offer info categories
cat_dict = {'x': 1}
offer_tab_en_x = []
for r in dfp['offer_tab']:
    try: 
        rec = eval(r)
    except:
        rec = ['']

    #print('\nrecord-->', rec[0] )
    if len(rec) > 2:
        for i in range( 2, len( rec ) ):
            if rec[i][0] in cat_dict:
                cat_dict[rec[i][0]] += 1                
            else:
                cat_dict[rec[i][0]] = 1                
    else:
        cat_dict['x'] += 1
        offer_tab_en_x.append(r)

print(cat_dict)
print(offer_tab_en_x)

df_offer_info_cat = pd.DataFrame(columns = ['category','count'])
df_offer_info_cat['category'] = [x for x in cat_dict.keys()]
df_offer_info_cat['count'] = [x for x in cat_dict.values()] 
df_offer_info_cat.to_excel('pracuj_analityk_w_kat_translated_1b_offer_req_dic.xlsx')


# Part 3
# offer_tab --> to category coluns

print(dfp.info())

df_offer = pd.DataFrame(columns = [x for x in cat_dict], index=dfp.index )
df_offer['href'] = ''

for i, r in dfp.iterrows():
    
    try: 
        rec = eval(r['offer_tab'])    
    except:
        rec = ['eval fail',['index '+str(i)]]
    
    #print(i,'\nrecord-->', rec[0] )
    df_offer.loc[ i, ['href'] ] = rec[0][0]
    if len(rec) > 2:
        for k in range( 2, len( rec ) ):
            #print(rec[k][0], rec[k][1])
            
            df_offer.loc[ i, rec[k][0] ] = str(rec[k][1]).replace('[','').replace(']','')
 #           'xxxx'.decode('unicode_escape').encode('ascii', 'ignore')
 #           print(df_offer.iloc[i])
    else:
        df_offer.loc[ i, 'x' ] = ''

print(df_offer.info())

dff = pd.concat([dfp,df_offer],axis=1)
dff.info()

dff.to_excel('pracuj_analityk_w_kat_translated_1c_offer_tab_in_columns.xlsx')



# Part 4

#df_offer = pd.read_excel('pracuj_analityk_w_kat_translated_1c_offer_tab_in_columns.xlsx')
#print(df_offer.info())


# create dataset 'responsibilities-1', 'requirements-1', 
#col_to_keep = ['responsibilities-1', 'requirements-1','offered-1',
#               'technologies-1', 'training-space-1', 'benefits-1']


#for c in col_to_keep:
#    print(c)
#    dfp[c] = translate_list(dff[c], verbose=False)
#    dfp.to_excel('pracuj_analityk_w_kat_translated_1d_dfp_dataset.xlsx')
#    print(dfp.info())


# Part 5

print('duplicates', dfp.duplicated(keep='first').sum(), len(dfp))
dfp = dfp.drop_duplicates()
print(len(dfp))

dfp = dfp.fillna('')
dfp.to_excel('pracuj_analityk_w_kat_translated_1d_dfp_dataset.xlsx')












