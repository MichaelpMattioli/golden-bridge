import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import pytz
import pandas as pd

class StoresExtractorSubway:
    def __init__(self):
        self.headers = {
            'accept-language': 'pt-BR,pt;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.41',
            'x-application-key': 'Bf8uR68TGUgRFf3DcPRY1wts'
        }
        self.subway_list = []
        self.df_lojas = self.scrape_subway()

    class UnidadeSubway:
        def __init__(self, display_order, store_id, store_name, district, city, state, address, latitude, longitude, web_address, phone_number, update_datetime):
            self.display_order = display_order
            self.store_id = store_id
            self.store_name = store_name
            self.district = district
            self.city = city
            self.state = state
            self.address = address
            self.latitude = latitude
            self.longitude = longitude
            self.web_address = web_address
            self.phone_number = phone_number
            self.update_datetime = update_datetime

    def scrape_subway(self):
        req = requests.get('https://restaurantes.subway.com/brasil', headers=self.headers)
        soup = BeautifulSoup(req.content, 'html.parser')

        for item in soup.find_all('div', attrs={'data-ordem': True}):
            display_order = item['data-ordem']
            store_id = item['data-codigo-unidade']
            store_name = item.find('h4').string
            district = item['data-bairros']
            city = item['data-cidades']
            state = item['data-estados']
            address = item['data-endereço']

            lat_long = re.search('destination=(.*)&travelmode', item.find('a', attrs={'data-acao': 'rota'})['href']).group(1).split(',')
            latitude = lat_long[0]
            longitude = lat_long[1]

            web_address = item.find('a')['href']
            phone_number = item.find('a', attrs={'data-acao': 'ligar'})['href'].replace("tel:", "")

            self.subway_list.append(self.UnidadeSubway(display_order, store_id, store_name, district, city, state, address, latitude, longitude, web_address, phone_number, datetime.now(tz=pytz.utc)))
        return pd.DataFrame(vars(s) for s in self.subway_list)