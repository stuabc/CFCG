import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def get_all_csv_name(path):
    filename_list = []
    for folderName, subfolders, filenames in os.walk(path):
        for file_name in filenames:
            if '.csv' in file_name:
                filename_list.append(file_name)
    return filename_list

pathgen = "./data/"  # this path contain the electricyt genneration for each countriy
targetpathgen = "./data/GEN" # outputp path

filename_gen = get_all_csv_name(pathgen)

soure_energy = ['Oil','Coal','Gas','Nuclear','Wind','Solar','Hydro','Geothermal','Biomass','Other']

for i in filename_gen:
    dataset_gen = pd.read_csv(pathgen +'/' + i[:-4] +'/'+i, header=0)
    dataset_gen.replace('n/e',0,inplace=True)
    dataset_gen.replace('#VALUE!',None,inplace=True)
    dataset_gen.fillna(method = 'ffill', inplace=True)
    dataset_gen.fillna(method = 'bfill', inplace=True)
    
    Total = np.zeros(len(dataset_gen))
    lce = np.zeros(len(dataset_gen))
    de = np.zeros(len(dataset_gen))
    for n in soure_energy:
        if n in dataset_gen.columns:
            if n == 'Oil':
                Total=Total + dataset_gen[n].values.astype(float)
                lce = lce + dataset_gen[n].values.astype(float)*650
                de = de + dataset_gen[n].values.astype(float)*406
            elif n == 'Coal':
                Total=Total + dataset_gen[n].values.astype(float)
                lce = lce + dataset_gen[n].values.astype(float)*820
                de = de + dataset_gen[n].values.astype(float)*760
            elif n == 'Gas':
                Total=Total + dataset_gen[n].values.astype(float)
                lce = lce + dataset_gen[n].values.astype(float)*490
                de = de + dataset_gen[n].values.astype(float)*370                
            elif n == 'Nuclear':
                Total=Total + dataset_gen[n].values.astype(float)
                lce = lce + dataset_gen[n].values.astype(float)*12
            elif n == 'Wind':
                Total=Total + dataset_gen[n].values.astype(float)
                lce = lce + dataset_gen[n].values.astype(float)*11
            elif n == 'Solar':
                Total=Total + dataset_gen[n].values.astype(float)
                lce = lce + dataset_gen[n].values.astype(float)*45       
            elif n == 'Hydro':
                Total=Total + dataset_gen[n].values.astype(float)
                lce = lce + dataset_gen[n].values.astype(float)*24       
            elif n == 'Geothermal':
                Total=Total + dataset_gen[n].values.astype(float)
                lce = lce + dataset_gen[n].values.astype(float)*38      
            elif n == 'Biomass':
                Total=Total + dataset_gen[n].values.astype(float)
                lce = lce + dataset_gen[n].values.astype(float)*230                 
            elif n == 'Other':
                Total=Total + dataset_gen[n].values.astype(float)
                lce = lce + dataset_gen[n].values.astype(float)*700
                de = de + dataset_gen[n].values.astype(float)*575  
    CI_de = de/Total
    CI_lce = lce/Total
    dataset_gen.insert(dataset_gen.shape[1], 'Total', Total)
    dataset_gen.insert(dataset_gen.shape[1], 'de', de)
    dataset_gen.insert(dataset_gen.shape[1], 'lce', lce)
    dataset_gen.insert(dataset_gen.shape[1], 'CI_de', CI_de)
    dataset_gen.insert(dataset_gen.shape[1], 'CI_lce', CI_lce)
    dataset_gen.to_csv(targetpathgen+'/'+i, index=False )

    

    
    
                
                
                
                
                
                
