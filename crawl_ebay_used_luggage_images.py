
import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm
import pandas as pd


# url = 'https://www.ebay.com/sch/i.html?_from=R40&_nkw=luggage&_sacat=0&rt=nc&LH_ItemCondition=3000'
page = 'https://www.ebay.com/sch/i.html?_from=R40&_nkw=luggage&_sacat=0&rt=nc&LH_ItemCondition=3000&_ipg=240'
get_urls = []

def save_urls_to_csv(image_urls):
   df = pd.DataFrame({"links": image_urls})
   df.to_csv("links.csv", index=False, encoding="utf-8")


n = 90
for i in range(1,n+1):
    page = 'https://www.ebay.com/sch/i.html?_from=R40&_nkw=luggage&_sacat=0&rt=nc&LH_ItemCondition=3000&_ipg=240&_pgn='
    r = requests.get(page)
    soup = BeautifulSoup(r.text, 'html.parser')
    products = soup.find_all('a')
    products_href = []
    for product in products:
        if product.get('data-interactions') is not None:
            if product.get('href').endswith('=3000'):
                products_href.append(product.get('href'))
                print(product.get('href'))
    print(len(products_href))
    for url in tqdm(products_href):
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        images = soup.find_all('img')
        sources = ['data-zoom-src', 'data-src']
        for img in images:
            for src in sources:
                get_url = img.get(src)
                if get_url is not None:
                    get_urls.append(get_url)
                    save_urls_to_csv(get_urls)

print(len(get_urls))

# url = 'https://www.ebay.com/itm/285424637354?epid=13033159615&hash=item4274a255aa%3Ag%3AxKsAAOSwE2dk1W3e&amdata=enc%3AAQAIAAAA4J2SgNV33Ack7XvI0uQlhk7XBOdOL8ATQO%2F6acqRQcSelxevaQv8tuyLJiuUTvMOLTLTTxPrgz5NtPus1Yn0WKPX0Lt3mG1htWBIVtVtiTlk0B6r%2BzmoxH96H%2Fmy2zQBP7sNUc1e%2FrfHmytLZwyjO%2BwbUZ%2F3hy8uDe03VvYX9361CSbs7fF4J8galMTp16GKKZlqbBvZ6gM05gr5xCBGVI5CpLA33mLU1Nh%2FZ65ONpA44PFZmvpOuq02RKd88FUsh8Qy7R%2Fu4NsE4%2FCYYYdshZ8vH7yJr7Q0dt%2FtZgS%2FeGry%7Ctkp%3ABFBM8uj-hb9i&LH_ItemCondition=3000'
# r = requests.get(url)
# soup = BeautifulSoup(r.text, 'html.parser')
# images = soup.find_all('img')
# sources = ['data-zoom-src', 'data-src']
# get_urls = []
# for img in images:
#     for src in sources:
#        get_url = img.get(src)
#        if get_url is not None:
#            get_urls.append(get_url)
# print(get_urls)