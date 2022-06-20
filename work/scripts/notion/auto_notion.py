from notion.client import NotionClient
from notion.block import Block

token = "1d56270b343c744e9ec1088f0a1998cb851e0468730d853e539f7f01af7270e59d69b36135251455749ea746ab47a15d20d3ab8d2b0512c7f9321de6782d8b311a6a9c6359534f2a8c4e93ebd122"
client = NotionClient(token_v2=token)

list_url = 'https://www.notion.so/97a57c1bd634487682d74c2cfe91b5d3?v=2d1e483cb575479999105055e0450a78'
collection_view = client.get_collection_view(list_url) 



new_row = collection_view.collection.add_row()

from datetime import datetime

today = datetime.today()
new_row.title = today.strftime('%d.%m.%Y')
#new_row.title = today.strftime('%d.%m.%Y')
