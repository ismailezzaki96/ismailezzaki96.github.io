import gkeepapi

import os
from notion_client import Client
from pprint import pprint
from notion_client import APIErrorCode, APIResponseError
import logging
from notion.client import NotionClient
from notion.block import BasicBlock



#notion = Client(auth="secret_d7d2Sz7C3yypFwqNDCla11yPulGSIGeoQRIbGJJqmxZ"   ,  log_level=logging.INFO,
                #)
#url = "https://api.notion.com/v1/pages/page_id"

json = {
    "properties": {
        "title": {
            "title": [{"type": "text", "text": {"content": "A note from your pals at Notion"}}]
        }
    },
    "children": [
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "text": [{"type": "text", "text": {"content": "You made this page using the Notion API. Pretty cool, huh? We hope you enjoy building with us."}}]
            }
        }
    ]
}

#my_page = notion.pages.update("37b0ce1e-0eb4-4625-8734-d13b2beeeef7", **json)

token = "1d56270b343c744e9ec1088f0a1998cb851e0468730d853e539f7f01af7270e59d69b36135251455749ea746ab47a15d20d3ab8d2b0512c7f9321de6782d8b311a6a9c6359534f2a8c4e93ebd122"



# Obtain the `token_v2` value by inspecting your browser 
# cookies on a logged-in (non-guest) session on Notion.so
client = NotionClient(token_v2=token)

# Replace this URL with the URL of the page you want to edit
page = client.get_block("https://www.notion.so/test-43ab641838d149e4b680dee50b49a7bc")

print("The old title is:", page.title)

# You can use Markdown! We convert on-the-fly 
# to Notion's internal formatted text data structure.
page.title = "The title has now changed, and has *live-updated* in the browser!"

newchild = page.children.add_new(BasicBlock, title="Something to get done")

#https://gkeepapi.readthedocs.io/en/latest/#getting-labels

keep = gkeepapi.Keep()
success = keep.login('ismailezzaki96@gmail.com', 'mhkmsbxpvohndjqk')

#gnotes = keep.find(labels=[keep.findLabel('App Ideas')])

#for note in gnotes:
#    print(note)

#note = keep.createNote('Todo', 'Eat breakfast')
#note.pinned = False
#note.color = gkeepapi.node.ColorValue.Red
#keep.sync()

















