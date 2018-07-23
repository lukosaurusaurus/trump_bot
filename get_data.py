import tensorflow as tf
import numpy as np
import json, requests
from requests_oauthlib import OAuth1
consumer_key = 'Mg6AhB22e74CJu04Nd96Wypxf'
consumer_secret = 'dB57griApIYn7MsTqdt0IZXqrovWeJjhPNPYAxGa9sVLVTUF3P'
access_token = '1019448500981911553-poPhTkGSUg3V44yyEBlVXVqbLv8K7x'
access_secret = 'SUkHjWcdkqFJCC7XXtaY8xbLwJ7epHumkGkQCEVC0e8DK'
base_url = 'https://api.twitter.com/1.1/'
user = 'realDonaldTrump'
params = {"screen_name":"realDonaldTrump",
        "count":200,
        "tweet_mode":"extended",
        "exclude_replies":"true",
        "include_rts":"false"}



def get_api_info(params):
    auth = OAuth1(consumer_key,consumer_secret,
              access_token,access_secret)
    request_url = 'statuses/user_timeline.json?'
    for num,param in enumerate(params):
        if not num == 0:
            request_url = request_url+"&"
        request_url = request_url+param+"="+str(params[param])

    response = requests.get(base_url+request_url, auth=auth)

    if response.status_code == 200:
        return json.loads(response.content.decode('utf-8'))
    else:
        return None

file = open("input.txt",'w')
total_statuses = []
for i in range(int(3000/params["count"])):
    statuses = get_api_info(params)
    if statuses:
        for num,status in enumerate(statuses):
            total_statuses.append(status['full_text'])
            if num == len(statuses) - 1:
                params["max_id"] = status['id'] - 1
for line in total_statuses:
    file.write(str(line.encode('ascii','ignore'))+str("\n"))
file.close()

print("Done!")
# if statuses is not None:
#     length = len(statuses[0]['full_text'])
#     for status in statuses:
#         if len(status['full_text']) > length:
#             length = len(status['full_text'])
#         print(status['id_str']+status['full_text'])
#     print("max "+str(length))
# else:
#     print('[!] Request Failed')