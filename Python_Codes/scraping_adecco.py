#!/usr/bin/env python
# coding: utf-8

# In[2]:


from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, unquote, quote
from bs4 import BeautifulSoup
import pandas as pd
pd.set_option('display.max_colwidth', -1)
import numpy as np
from w3lib.url import safe_url_string
import time
import re


# In[3]:


#List that contains all links
urlList = []

#finding all possible link in table
for pg in range(1,22):
    if pg<10:
        parse = "/salary-guide/2019/0000" + str(pg) + "/"
        html = urlopen('https://adecco.co.th/salary-guide/2019/0000' + str(pg))
        #print(html.getcode())
    else:
        parse = "/salary-guide/2019/000" + str(pg) + "/"
        html = urlopen('https://adecco.co.th/salary-guide/2019/000' + str(pg)) 
        #print(html.getcode())
        
    bsObj = BeautifulSoup(html, "lxml")
    tag_a = bsObj.find_all(["li"])
    
    for i in tag_a[:]:
        children = i.findChild("a" , recursive=False)
        url =  children.get('href')
        #print(url)
        #print(url,type(url))
        try:
            match = "^" + "(" + parse + ")" + "(.*)"
            #print(match)
            m = re.match(match ,url)
            if m :
                print(url)
                urlList.append("https://adecco.co.th" +str(url))
        except:
            pass
    #print("=====End of pg=====")


# In[4]:


len(urlList)


# In[56]:


urlList


# In[34]:


testurl = urlList[30]


# In[35]:


html = urlopen(testurl)
bsObj = BeautifulSoup(html, "lxml")
print(html.getcode())


# In[36]:


bsObj.find("h3").get_text()


# In[37]:


bsObj.find("p", {"class": "des-en"}).get_text()


# In[38]:


bsObj.find('table').find_all('td')


# In[10]:


tablelist = bsObj.find('table').find_all('td')


# In[16]:


first = (tablelist[5].get_text().strip())
second = (tablelist[6].get_text().strip())
third = (tablelist[7].get_text().strip())
print(first,second,third)


# In[41]:


counter = 0

urldict = dict()
jobdict = dict()
jddict = dict()
firstdict = dict()
seconddict = dict()
thirddict = dict()


# In[44]:


def splitstr(string):
    if string == "":
        return -1
    elif "-" not in string:
        value = int(string.replace(",",""))
        return value
    else:
        lower, upper = string.replace(",","").split("-") 
        lower = int(lower)
        upper = int(upper)
        value = (lower+upper)/2
        return value

import time

def findfrompg(pg):
    global urldict, jobdict, jddict, firstdict, seconddict, thirddict
    
    for url in urlList[pg:]:
        t0 = time.time()
        print(pg)
        html = urlopen(url)
        bsObj = BeautifulSoup(html, "lxml")
        urldict[pg] = url
        job = bsObj.find("h3").get_text().strip()
        jobdict[pg] = job
        print(job)
        jd = bsObj.find("p", {"class": "des-en"}).get_text()
        jddict[pg] = jd
        print(jd)
        tablist = bsObj.find('table').find_all('td')
        first = tablist[5].get_text().strip()
        second = tablist[6].get_text().strip()
        third = tablist[7].get_text().strip()

        firstdict[pg] = splitstr(first)
        seconddict[pg] = splitstr(second)
        thirddict[pg] = splitstr(third)
        t1 = time.time()
        print(t1-t0)
        
        pg += 1


# In[45]:


findfrompg(0)


# In[53]:


data_list = [jobdict, jddict, firstdict, seconddict, thirddict, urldict]
df = pd.DataFrame.from_dict(data_list, orient='columns').T
df.columns = ["position", "jobdesc", "newgrad", "junior", "senior", "url"]


# In[55]:


df


# In[69]:


len(df[(df.junior > 0)])


# In[65]:


len(df[(df.junior < 0)& (df.newgrad <0)])


# In[66]:


len(df[(df.junior < 0)& (df.newgrad > 0) & (df.senior<0)])


# In[71]:


s1 = (df[(df.junior > 0)]).junior
s2 = (df[(df.junior < 0)& (df.newgrad <0)]).senior
s3 = (df[(df.junior < 0)& (df.newgrad > 0) & (df.senior<0)]).newgrad


# In[74]:


salary = s1.append(s2).append(s3)


# In[78]:


salary = salary.sort_index(axis=0)


# In[77]:


salary.sort_index(axis=0)


# In[80]:


sal_df =pd.DataFrame(salary)


# In[84]:


sal_df


# In[86]:


df = df.join(sal_df)


# In[90]:


df.columns = ["position", "jobdesc", "newgrad", "junior", "senior", "url", "wage"]


# In[91]:


df.head()


# In[92]:


df.to_csv("adecco.csv")


# In[ ]:




