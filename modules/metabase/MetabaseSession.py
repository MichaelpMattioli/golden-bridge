import requests
class MetabaseSession:
    def __init__(self, username: str, password: str, remember: bool = True, base_url: str = 'https://adbikpi.arcosdorados.net/api/'):
        self.username = username
        self.password = password
        self.remember = remember
        self.base_url = base_url

    def login(self) -> str:
        headers_session = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.35',
        }

        json_data_session = {
            'username': self.username,
            'password': self.password,
            'remember': self.remember,
        }

        response_session = requests.post(f'{self.base_url}session', headers=headers_session, json=json_data_session, verify=False)
        session_id = response_session.cookies.get_dict()['metabase.SESSION']

        return session_id