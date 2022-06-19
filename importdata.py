import re
import os
import pandas as pd


def  import_data(path= "./data"):
    # regex = re.compile('.csv')
    D = []
    for root, dirs, files in os.walk(path):
        #print (root)
        #print(dirs)
        #print(files)
        for file in files:
            if re.search('csv', file):
                if len(D)==0:
                    D = pd.read_csv(os.path.join(path, file))
                else:
                    D = D.merge(pd.read_csv(os.path.join(path, file)), how = 'inner', on = 'id_men')
    return D