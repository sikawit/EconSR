#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
pd.set_option('display.max_colwidth', -1)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
import gensim
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
import nltk
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
from collections import Counter
import statsmodels.api as sm


# In[4]:


def calculatevec(l):
    v = np.zeros(300)
    counter = 0
    for word in l:
        if word not in stops.union("k"):
            try:
                vec = model.word_vec(word)
                v += vec
                counter +=1
            except:
                pass
    if counter == 0:
        return v
    else:
        return v/counter
def cleaning(rawtext):
    cleaned = re.sub("[^a-zA-Z]"," ", rawtext.lower())
    words = cleaned.split()
    return words
def series_to_vec(series):
    s = list(series.map(cleaning))
    n_pos = series.shape[0]
    L1 = np.array(calculatevec(s[0])).reshape(1,300)
    for i in range(1,n_pos):
        L2 = np.array(calculatevec(s[i])).reshape(1,300)
        L1 = np.concatenate((L1,L2))
    return L1
#L1 เป็น vector ของคำทั้งหมด

def cluster_kmeans(vector, nclus):
    kmeans = KMeans(n_clusters=nclus, random_state=0).fit(vector)
    kmeans.labels_
    k = pd.Series(kmeans.labels_)
    return k


def insert_k_to_df(df, series, colname):
    #df2 = df.copy()
    df.insert(loc = 0, column=str(colname), value=series)
    return df


def find_centroid(series, L1, k_num):
    group = series.values 
    #dec_g.k_pos.values
    no_pos = series.shape[0]
    centroid = np.zeros((k_num,300))
    njob = np.zeros((k_num,1))
    centroid = np.zeros((k_num,300))
    njob = np.zeros((k_num,1))
    for i in range(0,len(group)):
        njob[group[i]] += 1
        centroid[group[i]] = centroid[group[i]]+L1[i]
    for i in range(0,k_num):
        centroid[i] = centroid[i]/njob[i]
    return centroid

def centroid_2d(centroid):
    cen2d = TSNE(n_components=2, random_state=0).fit_transform(centroid)
    return cen2d

def plot_scatter(cen2d, title):
    x_plot, y_plot = cen2d.T
    plt.scatter(x_plot,y_plot)
    plt.title(title)
    return

def plot_clustergroup(cen2d, title, dict_commonword, dict_value, k_num):
    x_plot, y_plot = cen2d.T
    fig, ax = plt.subplots()
    scatter_x = x_plot
    scatter_y = y_plot
    clustergroup = np.arange(0,15) #[0,1,...,14]
    #most_common_list = list()
    
    #for g in np.unique(clustergroup):
        
        #i = np.where(clustergroup == g)
        #m = list_of_mostcommon[g]
    #ax.scatter(x_plot[i], y_plot[i], label=str(g)+str(m))
    
    l = list()
    for group in range(k_num):
        size = int(dict_value[group]/10)
        l.append(size)
    
    #print("type l", type(l))
    #print(l)
    
    #let s is the size of scatter point
    
    ax.scatter(x_plot, y_plot, s= l)
    
    sorted_dict = sorted(dict_commonword.keys())
    
    for group in range(len(sorted_dict)):
        g = str(group)
        w = dict_commonword[group]
        txt = str(g) + " " + str(w)
        ax.annotate(txt, (x_plot[group], y_plot[group]))
    
    #for k, txt in enumerate(list_of_mostcommon):
    #    ax.annotate(txt, (x_plot[k], y_plot[k]))
        
    fig.set_figheight(10)
    fig.set_figwidth(10)
    ax.legend()
    plt.suptitle(title)
    return

def find_most_common_pos_list(df, k_num):
    d = dict()
    
    for g in range(k_num):
        wcdf = df[df.k_pos == g]
        s = list(wcdf.pos_en.map(cleaning))
        flatlist = [item for sublist in s for item in sublist]
        c = [w for w in flatlist if not w in stops]
        most_com = most_common(c)
        d[g] = most_com

    return d

def find_most_common_jd_list(df, k_num):
    d = dict()
    
    for g in range(k_num):
        wcdf = df[df.k_jd == g]
        s = list(wcdf.jd_en.map(cleaning))
        flatlist = [item for sublist in s for item in sublist]
        c = [w for w in flatlist if not w in stops]
        most_com = most_common(c)
        d[g] = most_com

    return d

def most_common(lst):
    return max(set(lst), key=lst.count)

#####################3
def common_word(df, series):
    if(0 == 1):
        print("Most common word")
        for g in range(0,15):
            wcdf = df[series == g]
            s = list(wcdf.pos_en.map(cleaning))
            flatlist = [item for sublist in s for item in sublist]
            print("Group: " + str(g) + " " + most_common(flatlist))
        return

    print("Most common word")
    for g in range(0,15):
        #wcdf = dec_g[dec_g.k15_pos == g]
        s = df[series == g].pos_en.map(cleaning)
        flatlist = [item for sublist in s for item in sublist]
        print("Group: " + str(g) + " " + most_common(flatlist))
    return


def dendrogram(centroid_vector, titlename, labeldict):
    lab = list(labeldict.values())
    Z = hierarchy.linkage(centroid_vector, "single")
    plt.figure(figsize = (15,8))
    plt.title = str(titlename)
    dn = hierarchy.dendrogram(Z, leaf_rotation=30, labels = lab)
    return


def top_k(numbers, k=2):
    """The counter.most_common([k]) method works
    in the following way:
    >>> Counter('abracadabra').most_common(3)  
    [('a', 5), ('r', 2), ('b', 2)]
    """
    c = Counter(numbers)
    most_common = [key for key, val in c.most_common(k)]

    return most_common

def find_best_k(vec):
    for n_cluster in range(2, 15):
    #print(n_cluster)
        kmeans = KMeans(n_clusters=n_cluster).fit(X)
        label = kmeans.labels_
        sil_coeff = silhouette_score(X, label, metric='euclidean')
        print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
    return

def stats_k(k_series, dict_jobname):
    keys, vals = np.unique(k_series, return_counts=True)
    keys = pd.DataFrame(keys)
    vals = pd.DataFrame(vals)
    group_name = pd.DataFrame(list(dict_jobname.values()))
    stats = pd.merge(keys, group_name, left_index=True, right_index=True)
    stats = pd.merge(stats, vals,  left_index=True, right_index=True)
    stats.columns = ["group_num", "group_name", "val_count"]
    return stats


# In[5]:


def main_calc(df, k_num):
    vec_jd = series_to_vec(df.jd_en)
    k_jd = cluster_kmeans(vec_jd, k_num)
    df2 = df.copy()
    df2.insert(loc = 0, column=str("k_jd"), value=k_jd)
    #print("COMMON WORDS")    
    #for g in range(0,k_num):
        #wcdf = dec_g[dec_g.k15_pos == g]
    #    s = df2[df2.k_jd == g].jd_en.map(cleaning)
    #    flatlist = [item for sublist in s for item in sublist]
    #    c = [w for w in flatlist if not w in stops]
    #    top_freq = " ".join(top_k(c, k=10))
    #    print("Group: " + str(g) + " " + str(top_freq))
    return df2, vec_jd

def main_viz(df, vector, k_num, dict_name):
    centroid_jd = find_centroid(df.k_jd, vector, k_num)
    cen2d_jd = centroid_2d(centroid_jd)
    dendrogram(centroid_jd, "str(df)", dict_name)
    stats = stats_k(df.k_jd, dict_name)
    plot_clustergroup(cen2d_jd, "str(df)", dict_name, stats.val_count, k_num)
    #plt.pie(stats.val_count, labels = stats.group_name);
    #print(centroid_jd)
    return stats, centroid_jd

def main_viz_2(df, centroid_jd, k_num, dict_name):
    cen2d_jd = centroid_2d(centroid_jd)
    dendrogram(centroid_jd, "Title", dict_name)
    stats = stats_k(df.k_jd, dict_name)
    plot_clustergroup(cen2d_jd,"Title", dict_name, stats.val_count, k_num)
    #plt.pie(stats.val_count, labels = stats.group_name);
    return stats

def main_cluster(df):
    vec = series_to_vec(df.jd_en)
    c = vec-centroid_dec[0]
    d = np.linalg.norm(c, axis=1)
    df_dist =pd.DataFrame(d)
    for i in range(1,8):
        c = vec-centroid_dec[i]
        d = np.linalg.norm(c, axis=1)
        df_dist= pd.concat([df_dist, pd.DataFrame(d)], axis = 1, ignore_index=True)
    k_cluster = pd.DataFrame(df_dist.idxmin(axis=1))
    k_cluster.columns = ["k_jd"]
    x = np.where(k_cluster == 0)[0]
    clus = np.mean(vec[x], axis=0).reshape(1,300)
    for group in range(1,8):
        x = np.where(k_cluster == group)[0]
        clus_i = np.mean(vec[x], axis=0).reshape(1,300)
        print(clus_i.shape)
        clus = np.concatenate((clus, clus_i), axis=0)
    df_calc = pd.concat([k_cluster, df],axis =1)
    return df_calc, clus, vec

g= 3
wcdf = jan_fn[jan_fn.k_jd == g]
s = list(wcdf.jd_en.map(cleaning))
flatlist = [item for sublist in s for item in sublist]
c = [w for w in flatlist if not w in stops]
wc = WordCloud(font_path='THSarabunNew.ttf', # path ที่ตั้ง Font
            background_color="white", # ตั้งค่าพืสี้นหลัง
            regexp=r"[\u0E00-\u0E7Fa-zA-Z']+",
            width=1600, # กว้าง
            height=900).generate(" ".join(c))
#print("Group : " + str(g))
#plt.title("Test")
plt.figure(figsize=(20,10))
plt.imshow(wc)
plt.suptitle("Group:" + str(g) +  " "+dict_jd_jan_8[g])
plt.show()
# In[6]:


def wc(df,list_g,dict_name):
    for g in list_g:
        wcdf = df[df.k_jd == g]
        s = list(wcdf.jd_en.map(cleaning))
        flatlist = [item for sublist in s for item in sublist]
        c = [w for w in flatlist if not w in stops]
        wc = WordCloud(font_path='THSarabunNew.ttf', # path ที่ตั้ง Font
                    background_color="white", # ตั้งค่าพืสี้นหลัง
                    regexp=r"[\u0E00-\u0E7Fa-zA-Z']+",
                    width=1600, # กว้าง
                    height=900).generate(" ".join(c))
        #print("Group : " + str(g))
        #plt.title("Test")
        plt.figure(figsize=(20,10))
        plt.imshow(wc)
        plt.suptitle("Group:" + str(g) +  " "+dict_name[g])
        plt.show()
        
def topwords(df, list_g):
    for g in list_g:
            #wcdf = dec_g[dec_g.k15_pos == g]
            s = df[df.k_jd == g].jd_en.map(cleaning)
            flatlist = [item for sublist in s for item in sublist]
            c = [w for w in flatlist if not w in stops]
            top_freq = " ".join(top_k(c, k=10))
            print("Group " + str(g) + ": " + str(top_freq))


# In[7]:


#model
model = gensim.models.KeyedVectors.load_word2vec_format('/Users/sikawit/OneDrive/Chula/2018-2/Research/GoogleNews-vectors-negative300.bin', binary=True)


# In[8]:


dec_g = pd.read_csv("/Users/sikawit/OneDrive/Chula/2018-2/Research/Code_upload/g_dec2018.csv")
jan_g = pd.read_csv("/Users/sikawit/OneDrive/Chula/2018-2/Research/Code_upload/g_jan2019.csv")
feb_g = pd.read_csv("/Users/sikawit/OneDrive/Chula/2018-2/Research/Code_upload/g_feb2019.csv")
mar_g = pd.read_csv("/Users/sikawit/OneDrive/Chula/2018-2/Research/Code_upload/g_mar2019.csv")
adecco = pd.read_csv("/Users/sikawit/OneDrive/Chula/2018-2/Research/Code_upload/clustered_adecco.csv")
adecco.rename(columns={'jobdesc':'jd_en'}, inplace=True)


# In[9]:


dec_g.head()


# In[10]:


dec_calc, vec_dec = main_calc(dec_g,8)


# In[11]:


dict_jd_dec_8 = {0: "customers services", 1: "engineering", 2:"sales",
                3: "production", 4:"accounting and finance", 5:"language works",
                6: "office works", 7:"marketing"}
dec_calc.head()


# In[12]:


k_num =8
for g in range(0,k_num):
        #wcdf = dec_g[dec_g.k15_pos == g]
        s = dec_calc[dec_calc.k_jd == g].jd_en.map(cleaning)
        flatlist = [item for sublist in s for item in sublist]
        c = [w for w in flatlist if not w in stops]
        top_freq = " ".join(top_k(c, k=10))
        print("Group " + str(g) + ": " + str(top_freq))


# In[52]:


dict_jd_dec_8 = {0: "customers services", 1: "engineering", 2:"sales",
                3: "production", 4:"accounting and finance", 5:"language works",
                6: "office works", 7:"marketing"}


# In[53]:


dec_stats, centroid_dec = main_viz(dec_calc,vec_dec,8,dict_jd_dec_8)


# In[15]:


centroid_dec.shape


# In[16]:


#########################3


# In[17]:


vec_jan = series_to_vec(jan_g.jd_en)


# In[18]:


vec_jan


# In[19]:


centroid_dec


# In[58]:


c = vec_jan-centroid_dec[0]
d = np.linalg.norm(c, axis=1)
df_dist =pd.DataFrame(d)


# In[59]:


df_dist


# In[60]:


for i in range(1,8):
    c = vec_jan-centroid_dec[i]
    d = np.linalg.norm(c, axis=1)
    df_dist= pd.concat([df_dist, pd.DataFrame(d)], axis = 1, ignore_index=True)


# In[62]:


jan_dist =df_dist


# In[63]:


jan_dist


# In[71]:


jan_cluster = pd.DataFrame(jan_dist.idxmin(axis=1))
jan_cluster.columns = ["cluster"]


# In[73]:


jan_cluster.shape


# In[74]:


vec_jan.shape


# In[75]:


a = vec_jan
df_test = pd.DataFrame({"a": [a]})


# In[79]:


del df_test


# In[83]:


vec_jan


# In[84]:


vec_jan.shape


# In[87]:


jan_cluster


# In[92]:


x = np.where(jan_cluster ==0)[0]


# In[110]:


np.mean(vec_jan[x], axis=0)


# In[128]:


x = np.where(jan_cluster == 0)[0]
clus = np.mean(vec_jan[x], axis=0).reshape(1,300)


# In[132]:


for group in range(1,8):
    x = np.where(jan_cluster == group)[0]
    clus_i = np.mean(vec_jan[x], axis=0).reshape(1,300)
    print(clus_i.shape)
    
    clus = np.concatenate((clus, clus_i), axis=0)


# In[147]:


clus


# In[136]:


centroid_jan = clus


# In[139]:


jan_cluster.columns = ["k_jd"]


# In[140]:


jan_cluster


# In[146]:


jan_calc = pd.concat([jan_cluster, jan_g],axis =1)


# In[145]:





# In[176]:


def main_cluster(df):
    vec = series_to_vec(df.jd_en)
    c = vec-centroid_dec[0]
    d = np.linalg.norm(c, axis=1)
    df_dist =pd.DataFrame(d)
    for i in range(1,8):
        c = vec-centroid_dec[i]
        d = np.linalg.norm(c, axis=1)
        df_dist= pd.concat([df_dist, pd.DataFrame(d)], axis = 1, ignore_index=True)
    k_cluster = pd.DataFrame(df_dist.idxmin(axis=1))
    k_cluster.columns = ["k_jd"]
    x = np.where(k_cluster == 0)[0]
    clus = np.mean(vec[x], axis=0).reshape(1,300)
    for group in range(1,8):
        x = np.where(k_cluster == group)[0]
        clus_i = np.mean(vec[x], axis=0).reshape(1,300)
        print(clus_i.shape)
        clus = np.concatenate((clus, clus_i), axis=0)
    df_calc = pd.concat([k_cluster, df],axis =1)
    return df_calc, clus, vec


# In[ ]:


#############################################


# In[20]:


jan_calc, centroid_jan, vec_jan = main_cluster(jan_g)
feb_calc, centroid_feb, vec_feb = main_cluster(feb_g)
mar_calc, centroid_mar, vec_mar = main_cluster(mar_g)


# In[21]:


feb_calc.head()


# In[22]:


mar_calc


# In[168]:


vec.shape


# In[182]:


def main_viz_2(df, centroid_jd, k_num, dict_name):
    cen2d_jd = centroid_2d(centroid_jd)
    dendrogram(centroid_jd, "Title", dict_name)
    stats = stats_k(df.k_jd, dict_name)
    plot_clustergroup(cen2d_jd,"Title", dict_name, stats.val_count, k_num)
    #plt.pie(stats.val_count, labels = stats.group_name);
    return stats


# In[54]:


jan_stats = main_viz_2(jan_calc, centroid_jan, 8, dict_jd_dec_8)


# In[55]:


feb_stats = main_viz_2(feb_calc, centroid_feb, 8, dict_jd_dec_8)


# In[56]:


mar_stats = main_viz_2(mar_calc, centroid_mar, 8, dict_jd_dec_8)


# In[21]:


dec_g.head()


# In[57]:


wc(dec_calc,np.arange(0,8),dict_jd_dec_8)


# In[19]:


adecco.head()


# In[27]:


adecco_calc, centroid_adecco, vec_adecco = main_cluster(adecco)


# In[28]:


adecco_calc.head()


# In[29]:


centroid_adecco


# In[30]:


adecco_calc.head()


# In[31]:


sum_dist = np.sum(np.linalg.norm(vec_adecco - centroid_adecco[0],axis =1).reshape(1,766)+
np.linalg.norm(vec_adecco - centroid_adecco[1],axis =1).reshape(1,766)+
np.linalg.norm(vec_adecco - centroid_adecco[2],axis =1).reshape(1,766)+
np.linalg.norm(vec_adecco - centroid_adecco[3],axis =1).reshape(1,766)+
np.linalg.norm(vec_adecco - centroid_adecco[4],axis =1).reshape(1,766)+
np.linalg.norm(vec_adecco - centroid_adecco[5],axis =1).reshape(1,766)+
np.linalg.norm(vec_adecco - centroid_adecco[6],axis =1).reshape(1,766)+
np.linalg.norm(vec_adecco - centroid_adecco[7],axis =1).reshape(1,766), axis = 0)


# In[32]:


sum_dist.shape


# In[33]:


np.linalg.norm(vec_adecco - centroid_dec[0],axis=1)/sum_dist


# In[38]:


d_group0 = pd.DataFrame(-np.linalg.norm(vec_adecco - centroid_dec[0], axis = 1)/sum_dist)
d_group1 = pd.DataFrame(-np.linalg.norm(vec_adecco - centroid_dec[1], axis = 1)/sum_dist)
d_group2 = pd.DataFrame(-np.linalg.norm(vec_adecco - centroid_dec[2], axis = 1)/sum_dist)
d_group3 = pd.DataFrame(-np.linalg.norm(vec_adecco - centroid_dec[3], axis = 1)/sum_dist)
d_group4 = pd.DataFrame(-np.linalg.norm(vec_adecco - centroid_dec[4], axis = 1)/sum_dist)
d_group5 = pd.DataFrame(-np.linalg.norm(vec_adecco - centroid_dec[5], axis = 1)/sum_dist)
d_group6 = pd.DataFrame(-np.linalg.norm(vec_adecco - centroid_dec[6], axis = 1)/sum_dist)
d_group7 = pd.DataFrame(-np.linalg.norm(vec_adecco - centroid_dec[7], axis = 1)/sum_dist)
#d_sci = pd.DataFrame(np.linalg.norm(vec_adecco-sci, axis =1))

d_group0.columns = ["d_group0"]
d_group1.columns = ["d_group1"]
d_group2.columns = ["d_group2"]
d_group3.columns = ["d_group3"]
d_group4.columns = ["d_group4"]
d_group5.columns = ["d_group5"]
d_group6.columns = ["d_group6"]
d_group7.columns = ["d_group7"]
#d_sci.columns = ["d_sci"]


# In[39]:


normdist= pd.concat([d_group0,d_group1, d_group2, d_group3, d_group4, d_group5, d_group6, d_group7], axis=1)


# In[40]:


adecco_calc=pd.concat([adecco_calc, normdist], axis=1)


# In[41]:


adecco_calc.columns


# In[42]:


adecco_calc.head(5)


# In[44]:


adecco_calc = adecco_calc.rename(columns={"d_group0": "customers_services", "d_group1": "engineering",
                                          "d_group2":"sales","d_group3": "production", 
                                          "d_group4":"accounting_and_finance", "d_group5":"language_works",
                                          "d_group6": "office_works", "d_group7":"marketing",
                                          "marketings":"marketing", "value":"wage", "quality_control": "production"})


# In[45]:


adecco_calc.head()


# In[46]:


X = adecco_calc[["customers_services", "engineering", "sales", "production", "accounting_and_finance", 
                 "language_works", "office_works", "marketing", "newgrad", "senior"]]
X = sm.add_constant(X)
Y = np.log(adecco_calc['wage'])


# In[47]:


model_ols = sm.OLS(Y, X).fit()
predictions = model_ols.predict(X) 


# In[48]:


model_ols.summary()


# In[140]:


for g in range(8):
    print(str(g),np.mean(adecco_calc[adecco_calc.k_jd == g].wage),
          np.std(adecco_calc[adecco_calc.k_jd == g].wage),
          (adecco_calc[adecco_calc.k_jd == g].wage).shape)


# In[125]:


np.mean(adecco_calc.wage)


# In[49]:


adecco_stats = main_viz_2(adecco_calc, centroid_adecco, 8, dict_jd_dec_8)


# In[116]:


wc(adecco_calc,np.arange(0,8),dict_jd_dec_8)


# In[297]:


v1 =centroid_dec


# In[298]:


v2 =centroid_mar


# In[300]:


v1.shape


# In[301]:


v2.shape


# In[305]:


all_v = np.concatenate([v1,v2], axis =0)


# In[310]:


plt.scatter(all_v[:,0], all_v[:,1])


# In[315]:


plt.scatter(all_v[0,0], all_v[0,1])


# In[314]:


plt.scatter(all_v[8,0], all_v[8,1])


# In[316]:


(np.linalg.norm(vec_adecco - centroid_adecco[0], axis = 1))


# In[317]:


1/(np.linalg.norm(vec_adecco - centroid_adecco[0], axis = 1))


# In[46]:


print(dec_g.shape)
print(jan_g.shape)
print(feb_g.shape)
print(mar_g.shape)


# In[61]:


np.array(mar_stats.val_count)


# In[50]:


plt.pie(dec_stats.val_count, labels = dec_stats.group_name);


# In[80]:


all_v = np.concatenate([centroid_dec,centroid_mar], axis =0)


# In[66]:


all_v.shape


# In[67]:


plt.scatter(all_v[:,0], all_v[:,1])


# In[79]:


plt.scatter(centroid_dec[:,0],centroid_dec[:,1],color='red')
plt.scatter(centroid_jan[:,0],centroid_jan[:,1],color='blue')
plt.scatter(centroid_feb[:,0],centroid_feb[:,1],color='green')
plt.scatter(centroid_mar[:,0],centroid_mar[:,1],color='orange')
plt.figure(figsize=(10,10))
plt.show()


# In[82]:


v2d =TSNE(n_components=2, random_state=0).fit_transform(all_v)


# In[209]:


all_v = np.concatenate([centroid_dec,centroid_mar], axis =0)
plt.scatter(v2d[0,0],v2d[0,1],color='red')
plt.scatter(v2d[8,0],v2d[8,1],color='blue')


# In[86]:


TSNE(n_components=2, random_state=0).fit_transform(centroid_dec)


# In[87]:


TSNE(n_components=2, random_state=0).fit_transform(centroid_jan)


# In[104]:


norm_dec=np.linalg.norm(centroid_dec, axis =1).reshape(1,8)


# In[105]:


norm_jan=np.linalg.norm(centroid_jan, axis =1).reshape(1,8)


# In[106]:


norm_feb=np.linalg.norm(centroid_feb, axis =1).reshape(1,8)


# In[107]:


norm_mar=np.linalg.norm(centroid_mar, axis =1).reshape(1,8)


# In[108]:


norm_all = np.concatenate([norm_dec,norm_jan,norm_feb,norm_mar])


# In[110]:


norm_all[0]


# In[157]:


idx = np.arange(8)
u = list(dict_jd_dec_8.values())
print(u)


# In[112]:


idx


# In[210]:


plt.figure(figsize=[10,5])
axes = plt.gca()
axes.set_xlim([0.75,1.05])
#axes.set_ylim([ymin,ymax])
#plt.figure(figsize=[5,5],dpi=40)
c= ["red", "blue", "orange", "green"]
m = ["+", ".", "v", "^"]
for g in range(4):
    plt.scatter(norm_all[g][:],u,color=c[g], marker=m[g])
#plt.legend(('December 2018', 'March 2019'))
plt.gca().invert_yaxis()
#plt.title("KK")
plt.show()


# In[212]:


pd.DataFrame(norm_all)


# In[192]:


n =abs(norm_all[3] - norm_all[0])


# In[197]:


normdiff = pd.DataFrame([dict_jd_dec_8.values(),n]).T
normdiff.columns = ["cluster_name", "norm_difference"]


# In[201]:


(normdiff)


# In[219]:


np.linalg.norm(centroid_mar - centroid_dec[0], axis =1)


# In[222]:


np.concatenate([centroid_dec[0], centroid_mar])


# In[223]:


centroid_mar.shape


# In[224]:


centroid_dec[0].shape


# In[225]:


(centroid_mar - centroid_dec[0]).shape


# In[233]:


test_v =np.concatenate([centroid_mar,centroid_dec[0].reshape(1,300)])


# In[235]:


test_tsne =TSNE(n_components=2, random_state=0).fit_transform(test_v)


# In[238]:


test_v


# In[250]:


plt.scatter(test_tsne[8,0],test_tsne[8,1],color='red')
plt.scatter(test_tsne[1,0],test_tsne[1,1],color='blue')
plt.scatter(test_tsne[2:8,0],test_tsne[2:8,1],color='green')
plt.figure(figsize=(10,10))
plt.show()


# In[249]:


test_tsne[0,2]


# In[269]:


d0 = np.linalg.norm((vec_adecco - centroid_dec[0]), axis=1)
d1 = np.linalg.norm((vec_adecco - centroid_dec[1]), axis=1)
d2 = np.linalg.norm((vec_adecco - centroid_dec[2]), axis=1)
d3 = np.linalg.norm((vec_adecco - centroid_dec[3]), axis=1)
d4 = np.linalg.norm((vec_adecco - centroid_dec[4]), axis=1)
d5 = np.linalg.norm((vec_adecco - centroid_dec[5]), axis=1)
d6 = np.linalg.norm((vec_adecco - centroid_dec[6]), axis=1)
d7 = np.linalg.norm((vec_adecco - centroid_dec[7]), axis=1)


# In[270]:


sum_d =(d0+d1+d2+d3+d4+d5+d6+d7)


# In[64]:


sum_d


# In[274]:


d0[0]+d1[0]+d2[0]+d3[0]+d4[0]+d5[0]+d6[0]+d7[0]


# In[331]:


dist0 = np.log((d0/sum_d))
dist1 = np.log((d1/sum_d))
dist2 = np.log((d2/sum_d))
dist3 = np.log((d3/sum_d))
dist4 = np.log((d4/sum_d))
dist5 = np.log((d5/sum_d))
dist6 = np.log((d6/sum_d))
dist7 = np.log((d7/sum_d))


# In[332]:


distance_df = pd.DataFrame([dist0, dist1, dist2, dist3, dist4, dist5, dist6, dist7]).T
distance_df.columns = ["dist0", "dist1", "dist2", "dist3", "dist4", "dist5", "dist6", "dist7"]


# In[333]:


distance_df.head()


# In[334]:


adec_reg =adecco_calc[["wage","newgrad","senior"]]


# In[335]:


regn = pd.concat([adec_reg,distance_df], axis =1)


# In[ ]:





# In[338]:


X = regn[["dist0","dist2", "dist3", "dist5","dist7","newgrad","senior"]]
X = sm.add_constant(X)
Y = np.log(adecco_calc['wage'])


# In[339]:


model_ols = sm.OLS(Y, X).fit()
predictions = model_ols.predict(X) 
model_ols.summary()


# In[35]:


topwords(adecco_calc, range(8))


# In[33]:


wc(adecco_calc,[0,1],dict_jd_dec_8)


# In[59]:


adecco_stats =stats_k(adecco_calc.k_jd, dict_jd_dec_8)
adecco_stats.head()


# In[60]:


objects = adecco_stats.group_name
y_pos = np.arange(len(objects))
performance = adecco_stats.val_count


plt.figure(figsize=[10,5])
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation=45)

for a,b in zip(y_pos, performance):
    plt.text(a, b, str(b))

plt.show()


# In[63]:


topwords(adecco_calc, range(8))


# In[77]:


adecco_calc.iloc[0,:].T


# In[88]:


from math import pi

#https://python-graph-gallery.com/390-basic-radar-chart/
 
# Set data
df = pd.DataFrame({
'group': ['A','B','C','D'],
'customers_services': [1-(0.128904)**0.1, 0, 0, 0],
'engineering': [1-(0.143886)**0.1, 0, 0, 0],
'sales': [1-(0.132249)**0.1, 0, 0, 0],
'production': [1-(0.114645)**0.1, 0, 0, 0],
'accounting_and_finance': [1-(0.0805651 )**0.1 , 0, 0, 0],
"language_works": [1-(0.140878)**0.1,0,0,0],
"office_works": [1-(0.111435)**0.1,0,0,0] ,
"marketing":[1-(0.117172)**0.1,0,0,0]
})
 
# number of variable
categories=list(df)[1:]
N = len(categories)
 
# We are going to plot the first line of the data frame.
# But we need to repeat the first value to close the circular graph:
values=df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
values
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
plt.suptitle("Accounting Analyst")

# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='black', size=10)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.1,0.2,0.3], ["1","2","3"], color="grey", size=7)
plt.ylim(0,.3)
 
# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')
 
# Fill area
ax.fill(angles, values, 'b', alpha=0.1)



# In[89]:


adecco_calc.iloc[54,:].T


# In[33]:


from math import pi

#https://python-graph-gallery.com/390-basic-radar-chart/
 
# Set data
df = pd.DataFrame({
'group': ['A','B','C','D'],
'customers_services': [1-(0.136241)**0.01, 0, 0, 0],
'engineering': [1-(0.125052)**0.01, 0, 0, 0],
'sales': [1-(0.123129 )**0.01, 0, 0, 0],
'production': [1-(0.11446)**0.01, 0, 0, 0],
'accounting_and_finance': [1-(0.140312 )**0.01 , 0, 0, 0],
"language_works": [1-(0.125966)**0.01,0,0,0],
"office_works": [1-(0.130962)**0.01,0,0,0] ,
"marketing":[1-(0.123311)**0.01,0,0,0]
})
 
# number of variable
categories=list(df)[1:]
N = len(categories)
 
# We are going to plot the first line of the data frame.
# But we need to repeat the first value to close the circular graph:
values=df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
values
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
plt.suptitle("System Engineer")

# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='black', size=10)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.01,0.02,0.03], ["1","2","3"], color="grey", size=7)
plt.ylim(0,.03)
 
# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')
 
# Fill area
ax.fill(angles, values, 'b', alpha=0.1)




# In[ ]:




