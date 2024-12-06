from datetime import datetime
from elasticsearch import Elasticsearch
import pandas as pd 
from elasticsearch import helpers

# step 0: setting of elasticsearch and read data
client = Elasticsearch("http://localhost:9200/")
index_name = "data"
df = pd.read_csv('data.csv')
# step 1: preprocess dataframe 
df['About'] = df['About'].fillna('')
df['Voted'] = df['Voted'].fillna(0)
df = df.to_dict('records')
# some function need in continuation
def generate_docs(df):
    '''
        This function procces dataframe to a document format 
        input: dataframe
        filed About use drop_stop_words function  drop stop words and replace str with list of str
        output: This function use "for" to yield any row of dataframe ( convert to doc format)
    '''
    for c,line in enumerate(df):

        yield {
            "_index": index_name,
            "_id": c , 
            "_source": {
                "Name":line.get("Name",''),
                "Actors":line.get("Actors",''),
                "Score":line.get("Source",None),
                "About":drop_stop_words('rebuilt_persian',line.get("About",'')),
                "Genre":line.get("Genre",''),
                "Image":line.get("Image",''),
                "Crew":line.get("Crew",''),
                "Voted":line.get("Voted",None),
                "Type":line.get("Type",''),
            },
            
            }

def drop_stop_words(analyzer_name,text):
    '''
        This function drop stop words from text and convert to a list of str
        this function use analyzer from Elasticsearch(we need define analyzer in create index step)
        input: get analyzer_name and text
        output: list of str or token
    '''
    analizer_output = client.indices.analyze(index=index_name,body={
        "analyzer": analyzer_name,
        "text": text
    })
    final_text=''
    for token in analizer_output["tokens"]:
        final_text +=token['token'] + ' '
    return final_text

# step 2: create index with some setting from "setting_body"
setting_body = {
    "settings": {
        "analysis": {
            "char_filter": {
                "zero_width_spaces": {
                    "type":       "mapping",
                    "mappings": [ "\\u200C=>\\u0020"] 
                }
            },
            "filter": {
                "persian_stop": {
                "type":       "stop",
                "stopwords":  "_persian_" 
                }
            },
            "analyzer": {
                "rebuilt_persian": {
                    "tokenizer":     "standard",
                    "char_filter": [ "zero_width_spaces" ],
                    "filter": [
                        "lowercase",
                        "decimal_digit",
                        "arabic_normalization",
                        "persian_normalization",
                        "persian_stop"
                    ]
                }
            }
        }
    }
}

create_index=client.indices.create(index = index_name, ignore = [401,404],body = setting_body)

# step 3: asign any row of dataframe with some procces to index (we created in step 2)
custom = helpers.bulk(client, generate_docs(df))

