#!/usr/bin/env python
# coding: utf-8

# In[11]:


from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, unquote, quote
from bs4 import BeautifulSoup
import pandas as pd
pd.set_option('display.max_colwidth', -1)
import numpy as np

from w3lib.url import safe_url_string
import time


# In[12]:


posList = list() #keep html
urlList = dict() #keep url
jobList = dict() # keep position name
firmList = dict() #keep firm name
salList = dict() # keep salary
eduList = dict() # keep edu
locList = dict() #keep location


# In[13]:


def striptext(x):
    u = x.get_text().replace('\t','').strip()
    return u


# In[15]:


def findFromPg(pg):
    global posList, urlList, jobList, salList, eduList, firmList, locList, df, counter
    pg = int(pg)
    t0 = time.time()
    counter = 0
    while True:
        #if pg == 2: break
        print("pg = " , pg)
        html = urlopen('https://www.jobtopgun.com/AjaxServ?pg='+str(pg))
        bsObj = BeautifulSoup(html, "lxml")
        check = bsObj.find_all('div', {'class':'jobListPositionName'})
        if len(check) == 0:
            break
        posList += bsObj.findAll('div',{'class':'jobListPositionName'})
        for posName in check:
            #pathTh = unquote(posName.find('a')['href']).replace(' ',"%20")
            path = posName.find('a')['href'].replace(' ',"%20")
            #path = quote(posName.find('a')['href'])
            url = 'https://www.jobtopgun.com' + path
            #urlTh = 'https://www.jobtopgun.com' + pathTh
            #urlList.append(url)
            #urlThList.append(urlTh)

            firmName = posName.find_next('div')
            salName = firmName.find_next('div')
            eduName = salName.find_next('div')
            locName = eduName.find_next('div')

            pos = posName.find('a').get_text().strip()
            
            jobList[counter] = pos
            firmList[counter] = (striptext(firmName))
            salList[counter] = (striptext(salName))
            eduList[counter] = (striptext(eduName))
            locList[counter] = (striptext(locName))

            safeUrl = safe_url_string(url, encoding="utf-8")
            urlList[counter] = (safeUrl)
            
            t1 = time.time()
            print(t1-t0, pg)
            counter += 1
    
        pg += 1
        
    t1 = time.time()
    print(t1-t0, pg, counter)
        
    d = [jobList, firmList, salList, eduList, locList, urlList]
    
    pd.set_option('display.max_colwidth', -1)
    df = pd.DataFrame.from_dict(d)
    return df


# In[16]:


findFromPg(1)


# In[1]:


len(firmList)


# In[22]:


firmdf = pd.DataFrame(list(firmList.items()), columns=['idx', "company"])
jobdf = pd.DataFrame(list(jobList.items()), columns= ['idx', "position"])
saldf = pd.DataFrame(list(salList.items()), columns= ["idx", "salary"])
edudf = pd.DataFrame(list(eduList.items()), columns=["idx", "education"])
locdf = pd.DataFrame(list(locList.items()), columns= ["idx", "location"])
urldf = pd.DataFrame(list(urlList.items()), columns= ['idx', "url"])


# In[30]:


df1 = pd.merge(firmdf, jobdf, on="idx", how = "left")
df1 = pd.merge(df1, saldf, on="idx", how = "left")
df1 = pd.merge(df1, edudf, on="idx", how = "left")
df1 = pd.merge(df1, locdf, on="idx", how = "left")
df1 = pd.merge(df1, urldf, on="idx", how = "left")


# In[31]:


df1


# In[37]:


df1.salary = df1.salary.str.replace("เงินเดือน  : \n\r\n\r\n\r\n"," ")


# In[40]:


df1.education = df1.education.str.replace("วุฒิการศึกษา  : \n\r\n\r\n\r\n","")


# In[43]:


df1.location =df1.location.str.replace("สถานที่  : \n\r\n\r\n\r\n","")


# In[46]:


df1.salary = df1.salary.str.strip()


# In[47]:


df1.to_csv("df1.csv")


# In[48]:


df1.to_pickle("df1pkl.pkl")


# In[49]:


df1.head()


# In[52]:


df1.url[667]


# In[ ]:




