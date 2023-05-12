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

pathgen = "./data/GEN"     #This path contain local carbon intensity
pathtrans = "./data/FLOW"  #This path contain electricity flow
filename_gen = get_all_csv_name(pathgen)
filename_trans = get_all_csv_name(pathtrans)
num_time = 26304   #3 years

country = ['Austria (AT)', 'Belgium (BE)','Bulgaria (BG)','Switzerland (CH)','Cyprus (CY)', 
          'Czech Republic (CZ)','Germany (DE)','Denmark (DK)','Estonia (EE)','Spain (ES)','Finland (FI)','France (FR)',
          'Greece (GR)','Croatia (HR)','Hungary (HU)','Ireland (IE)','Italy (IT)','Lithuania (LT)','Latvia (LV)',
         'Netherlands (NL)','Norway (NO)','Poland (PL)','Portugal (PT)','Serbia (RS)','Romania (RO)',
          'Sweden (SE)','Slovenia (SI)','Slovakia (SK)']

num_country = len(country)


transfer = np.zeros((num_country,num_country,num_time)) 
generation = np.zeros((num_time,num_country)) 
total_power = np.zeros((num_time,num_country))
x =  np.zeros((num_country,num_country,num_time))
B =  np.zeros((num_country,num_country,num_time))
ep = np.zeros((num_time,num_country)) 
ex = np.zeros((num_time,num_country)) 
ci_network = np.zeros((num_time,num_country)) 
ci_direct = np.zeros((num_time,num_country)) 


#Get generation(production) matrix and direct carbon intensity matrix
for i in filename_gen:
    dataset_gen = pd.read_csv(pathgen +'/'+ i, header=0, low_memory=False)
    dataset_gen.replace('n/e',None,inplace=True)
    dataset_gen.replace('#VALUE!',None,inplace=True)
    dataset_gen.fillna(method = 'ffill', inplace=True)
    dataset_gen.fillna(method = 'bfill', inplace=True)
    index = country.index(dataset_gen['Area'][0])
    generation[:,index] = dataset_gen['Total']
    ci_direct[:,index] = dataset_gen['CI_lce']
for i in filename_trans:
    dataset_trans = pd.read_csv(pathtrans +'/' + i, header=0, low_memory=False)
    dataset_trans.replace('n/e',0,inplace=True)
    dataset_trans.replace('#VALUE!',None,inplace=True)
    dataset_trans.fillna(method = 'ffill', inplace=True)
    dataset_trans.fillna(method = 'bfill', inplace=True)  
    indexsource = country.index(list(dataset_trans)[1][0:list(dataset_trans)[1].index('>')-1])
    indextarget = country.index(list(dataset_trans)[1][list(dataset_trans)[1].index('>')+2:len(list(dataset_trans)[1])-5])
    transfer[indexsource,indextarget,:] = dataset_trans.iloc[:,1]
    indexsource = country.index(list(dataset_trans)[2][0:list(dataset_trans)[2].index('>')-1])
    indextarget = country.index(list(dataset_trans)[2][list(dataset_trans)[2].index('>')+2:len(list(dataset_trans)[2])-5])
    transfer[indexsource,indextarget,:] = dataset_trans.iloc[:,2]
    
ep = generation*ci_direct  # total emission / kg

# get actual total electricity
total_power = generation
for i in range(num_country):
    for j in range(num_time):
        total_power[j,i] = total_power[j,i] + sum(transfer[:,i,j])
        #total_power[j,i] = total_power[j,i] - sum(transfer[i,:,j])
for i in range(num_time) :
    for m in range(num_country):
        for n in range(num_country):
            B[m,n,i] = transfer[m,n,i]/total_power[i,m]   
for i in range(num_time):
    ex[i,:]=ep[i,:].dot(np.linalg.inv(np.identity(num_country)-B[:,:,i]))
     
for i in range(num_time):
    for m in range(num_country):
        ci_network[i,m] = ex[i,m]/total_power[i,m]

# df_ci_network = pd.DataFrame(ci_network,columns=country,index=dataset_gen['MTU'])
#df_ci_network.to_csv("ci_network_dirctef.csv",sep=',',index=True,header=True) 
#np.savetxt("ci_network_direct.csv",df_ci_network,delimiter=',')
