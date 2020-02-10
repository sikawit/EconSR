#!/usr/bin/env python
# coding: utf-8

# In[1]:


from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, unquote, quote
from bs4 import BeautifulSoup
import pandas as pd
pd.set_option('display.max_colwidth', -1)
import numpy as np
from w3lib.url import safe_url_string
import time


# In[2]:


jdDict= dict()
qDict= dict()
reqDict= dict()
firmtypeDict = dict()
postdateDict= dict()


# In[3]:


def scrape_web(i):
    global jdDict, qDict, reqDict, firmtypeDict , postdateDict
    
    url = df.url[i]
    try:
        html = urlopen(url+ str('/2'))
        checker = True
    except:
        checker = False
        jdDict[i] = np.nan
        qDict[i] = np.nan
        reqDict[i] = np.nan
        firmtypeDict[i] = np.nan
        postdateDict[i] = np.nan
    

    if checker :
        bsObj = BeautifulSoup(html.read(),"lxml")
        try:
            jd = bsObj.find('div',{'id':'jobDescriptionWithHeader'}).getText()
            jdDict[i] = jd
        except:
            jdDict[i] = np.nan

        try:
            q = bsObj.find('div',{'id':'qualificationWithHeader'}).getText()
            qDict[i] = q
        except:
            qDict[i] = np.nan

        try:
            reqList = bsObj.find('div',{'class':'row', 'align':'left'}).find('div').findAll('span')
            req = ''
            for r in reqList:
                req += r.getText().strip()
            reqDict[i] = req
        except:
            reqDict[i] = np.nan

        try:
            firmtype = bsObj.find('div',{'id':'industryName'}).find('span').get_text().strip()
            firmtypeDict[i]= firmtype
        except:
            firmtypeDict[i] = np.nan

        try:
            postdate = bsObj.find('td', {'style':"padding-left:10px"}).get_text()
            postdateDict[i] = postdate
        except:
            postdateDict[i] = np.nan



# In[5]:


df= pd.read_csv('/Users/sikawit/OneDrive/Chula/2018-2/Research/Jan2019/df1.csv')


# In[6]:


for pg in range(len(df)):
    t0 = time.time()
    url = df.url[pg]
    scrape_web(pg)
    t1 = time.time()
    print(pg, (t1-t0))


# In[12]:


jdDf = pd.DataFrame.from_dict(jdDict, orient='index')
qDf = pd.DataFrame.from_dict(qDict, orient='index')
reqDf = pd.DataFrame.from_dict(reqDict, orient='index')
firmtypeDf = pd.DataFrame.from_dict(firmtypeDict, orient='index')
postdateDictDf = pd.DataFrame.from_dict(postdateDict, orient='index')


# In[15]:


jdDf = jdDf.reset_index()


# In[16]:


jdDf.columns = ['idx', 'jobDesc']


# In[17]:


jdDf.head()


# In[28]:


def idx(df):
    df = df.reset_index()
    df.columns = ['idx', str(df)]
    return df


# In[18]:


qDf = qDf.reset_index()
qDf.columns = ['idx', 'qualification']


# In[19]:


qDf.head()


# In[20]:


reqDf = reqDf.reset_index()
reqDf.columns = ['idx', 'requirement']


# In[35]:


reqDf.head()


# In[21]:


firmtypeDf = firmtypeDf.reset_index()
firmtypeDf.columns = ['idx', 'type']


# In[37]:


firmtypeDf.head()


# In[22]:


postdateDictDf = postdateDictDf.reset_index()
postdateDictDf.columns = ['idx', 'date']


# In[39]:


postdateDictDf.head()


# In[23]:


df2 = pd.merge(jdDf,qDf,how = 'left')


# In[24]:


df2 = pd.merge(df2, reqDf, how = 'left')


# In[25]:


df2 = pd.merge(df2, firmtypeDf, how = 'left')


# In[26]:


df2 = pd.merge(df2, postdateDictDf, how = 'left')


# In[27]:


len(df2)


# In[28]:


df2.head()


# In[48]:


df.head()


# In[50]:


print(df.columns)


# In[51]:


df = df.rename(columns={'Unnamed: 0': 'idx'})


# In[52]:


df


# In[53]:


alldata = pd.merge(df,df2, how= 'left')


# In[56]:


alldata.to_csv('alldata.csv')


# In[59]:


df2.to_csv('part2.csv')


# In[61]:


df3 = df2.drop('idx', axis =1)


# In[68]:


df4 = df3.dropna(how = 'all')


# In[67]:


df.head()


# In[74]:


df4 = df4.reset_index()


# In[76]:


df4 = df4.rename({'index':'idx'})


# In[77]:


df4.to_csv('df4.csv')


# In[29]:


df2.head()


# In[32]:


df2.jobDesc


# In[35]:


def striptext(series):
    series = series.str.replace("\n","")
    series = series.str.replace("\t","")
    series = series.str.replace("\r","")
    return series


# In[36]:


df2.jobDesc = striptext(df2.jobDesc)
df2.qualification = striptext(df2.qualification)
df2.requirement = striptext(df2.requirement)


# In[37]:


df2.head()


# In[41]:


df2.date = df2.date.str.replace("/2562","/2019")
df2.date = df2.date.str.replace("/2561","/2018")


# In[42]:


df2.head()


# In[43]:


df2.date.value_counts()


# In[45]:


df2.head()


# In[47]:


df.head()


# In[48]:


df_all = pd.merge(df, df2, on='idx', how= 'left')


# In[51]:


df_all.type = striptext(df_all.type)


# In[52]:


df_all.type


# In[53]:


df_all.to_csv("df_all.csv")


# In[54]:


df_all.to_pickle("df_all.pkl")


# In[1]:


#เลือกบาง column มาใช้
def selectsomecolumn(df):
    df = df[["idx","position","jobDesc"]]
    df.jobDesc = df.jobDesc.str.replace("หน้าที่และความรับผิดชอบ","")
    print(len(df))
    df.jobDesc = df.jobDesc.str.strip()
    df =df.replace("",np.nan)
    df= df.dropna()
    print(len(df))
    return df


# In[ ]:




