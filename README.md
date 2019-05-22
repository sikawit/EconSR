# The tracking skills of Thai labour market using unsupervised machine learning with internet data

This repository is a part of my senior research on the Bachelor of Economics, Chulalongkorn University, Bangkok, Thailand.

In this project, I write the python scripts for

- Scraping data from job portal websites
- Vectorising the job descriptions by Word2Vec
- Clustering job descriptions
- Visualising the clustering result
- Predicting the wage from the skill set in job description vectors

I also upload documentations (paper and keynote files) in this repository.

## Setup

My codes are written in Python 3. To run my files, I recommend to use Jupyter Notebook. The required library are `numpy, pandas, BeautifulSoup, re, urllib, time, w3lib, math, gensim, wordcloud, nltk, collections, statsmodel, scipy, sklearn`.

## Usage

This project made of the following parts:

- Jobtopgun Data are scraped by `Part1_Collecting_URL_JTG.ipynb` for scarping url in JTG and `Part2_Scraping_URL_JTG.ipynb` for scraping each page in Jobtopgun.
- Adecco Data are scraped by `Scraping_Adecco.ipynb` and cleaned by `Cleaning_Adecco`.
- Translating, due to the limitation of Google Translate API, I use `googletranslate` function on Google Sheets to translate Thai into English. This process not in the Python script.
- Analysing, all analysing parts are in `Analysing.ipynb`.

Dataset are in `Jobtopgun_Data` and `Adecco_Data`.

Documentations are in `Documentations`.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Acknowledgement

I would like to thank
 - Tanapong Potipiti, Ph.D, Assoc. Prof., my advisor, who always advises and solving many technical problems in a lot of parts in this paper
 - Wasawat Somno, ThoughtWorks Thailand, who advises and inspires me in computer programming
-  My family who always trust me for 4 years at the Faculty of Economics, Chulalongkorn University
-  My friends who always support me while writing this paper


## License
Copyright (C) 2019 Sikkawit Amornnorachai Licensed under the [MIT Licence](https://choosealicense.com/licenses/mit/) (See the `Licence` file)

