import json
# import multiprocessing
# from multiprocessing import Pool
from feature import Features

class FeatureExtractor:

    def __init__(self):

        self.f = Features()

    '''
    Takes features of four datasets:
        i. google_3.json
        ii. fever_sup.json
        iii. fever_rej.json
        iv. fever_3.json
    and return dictionary in the form of (training_example, label)
    
    Note: In some datasets information such as triples/sentences are missing, 
        so we neglect them
    '''
    
    def get_features(self, data):

        train_data = dict()
        label = dict()
    
        for key, value in data.items():

            
            for i in range(len(data[key])):
                # if 1st iteration, intialize keys with values else 
                # append exising keys
                if i == 0:

                    if key == 'google_3.json':

                        train_data[str(key)] = [self.f.extract_features(value[i]['body'], value[i]['spo'][0],value[i]['spo'][1], value[i]['spo'][2])]
                        label[str(key)] = [value[i]['label']]

                    else:

                        train_data[str(key)] = [self.f.extract_features(value[i]['sentence'], value[i]['triples'][0][0],value[i]['triples'][0][1], value[i]['triples'][0][2])]
                        # in fever_sup, id of label is represented as "lablel"
                        if key == "fever_sup.json":
                            label[str(key)] = [value[i]['lablel']]
                        else:
                            label[str(key)] = [value[i]['label']]
                else:

                    if key == 'google_3.json':

                        train_data[str(key)].append(self.f.extract_features(value[i]['body'], value[i]['spo'][0],value[i]['spo'][1], value[i]['spo'][2]))
                        label[str(key)].append(value[i]['label'])

                    else:

                        # try is needed because information in few examples is missing
                        try:

                            train_data[str(key)].append(self.f.extract_features(value[i]['sentence'], value[i]['triples'][0][0],value[i]['triples'][0][1], value[i]['triples'][0][2]))
                            # in fever_sup, id of label is represented as "lablel"
                            if key == "fever_sup.json":
                                    label[str(key)].append(value[i]['lablel'])
                            else:
                                    label[str(key)].append(value[i]['label'])
                        except:
                            continue

        return train_data, label

        
    '''
    Load datafiles and store in the form of dictionary
    '''
    
    def load_datafiles(self, dataset_params):
    
        data = dict()
        for p in dataset_params:
                       
            open_data = open(p['EXP_FOLDER'] + p['DATASET'])
            data[str(p['DATASET'])] = json.load(open_data)
                    
        return data

if __name__ == '__main__':

    '''
        automatically extracts all features from a given dataset
    '''

    dataset_path = [{'EXP_FOLDER': '../../data/', 'DATASET': 'google_3.json'},   
                    {'EXP_FOLDER': '../../data/fever/reject/', 'DATASET': 'fever_rej.json'},
                     {'EXP_FOLDER': '../../data/fever/support/', 'DATASET': 'fever_sup.json'},
                     {'EXP_FOLDER': '../../data/fever/3-class/', 'DATASET': 'fever_3.json'}]

   
    featureextractor = FeatureExtractor()
       
    data = featureextractor.load_datafiles(dataset_path)
    
    #train_data and labels are dictionaries
    #each dictionary has key with dataset name
    train_data, label = featureextractor.get_features(data)
    