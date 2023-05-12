import pandas as pd
import numpy as np
import os
import typing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from utiliz import get_all_csv_name,create_tf_dataset,enconding_process
from model import GraphInfo,GraphConv,CFCG

ci_net = pd.read_csv("D:/ci_network.csv") 
# this path contains the carbon intensity of all nodes. (times, nodes)
ci_array = ci_net[ci_net.columns[1:]].values
adjacency_matrix = pd.read_csv("d:/Adj.csv",header=None).values
# this path contains adjacency matrix. (nodes, nodes)
pathtrans = "C:/poster/FLOW"
# this path contains electricity flows
time_steps_hours = ci_array.shape[0]
train_size, val_size = 0.6, 0.2
batch_size = 64
input_sequence_length = 24
forecast_horizon = 24
multi_horizon = True
in_feat = 5
out_feat = 5
lstm_units = 32 #64
epochs = 2

country = ['Austria (AT)', 'Belgium (BE)','Bulgaria (BG)','Switzerland (CH)','Cyprus (CY)', 
          'Czech Republic (CZ)','Germany (DE)','Denmark (DK)','Estonia (EE)','Spain (ES)','Finland (FI)','France (FR)',
          'Greece (GR)','Croatia (HR)','Hungary (HU)','Ireland (IE)','Italy (IT)','Lithuania (LT)','Latvia (LV)',
          'Netherlands (NL)','Norway (NO)','Poland (PL)','Portugal (PT)','Serbia (RS)','Romania (RO)',
          'Sweden (SE)','Slovenia (SI)','Slovakia (SK)']
num_nodes = len(country)


##transfer flows
filename_trans = get_all_csv_name(pathtrans)
transfer = np.zeros((time_steps_hours,num_nodes,num_nodes)) 
for i in filename_trans:
    dataset_trans = pd.read_csv(pathtrans +'/' + i, header=0, low_memory=False)
    dataset_trans.replace('n/e',0,inplace=True)
    dataset_trans.replace('#VALUE!',None,inplace=True)
    dataset_trans.fillna(method = 'ffill', inplace=True)
    dataset_trans.fillna(method = 'bfill', inplace=True)
    indexsource = country.index(list(dataset_trans)[1][0:list(dataset_trans)[1].index('>')-1])
    indextarget = country.index(list(dataset_trans)[1][list(dataset_trans)[1].index('>')+2:len(list(dataset_trans)[1])-5])
    transfer[:,indexsource,indextarget] = dataset_trans.iloc[:,1]
    indexsource = country.index(list(dataset_trans)[2][0:list(dataset_trans)[2].index('>')-1])
    indextarget = country.index(list(dataset_trans)[2][list(dataset_trans)[2].index('>')+2:len(list(dataset_trans)[2])-5])
    transfer[:,indexsource,indextarget] = dataset_trans.iloc[:,2]


train_array, val_array, test_array,mean,std = enconding_process(ci_array, transfer, train_size, val_size)

train_dataset, val_dataset = (
    create_tf_dataset(data_array, input_sequence_length, forecast_horizon, batch_size)
    for data_array in [train_array, val_array]
)

test_dataset = create_tf_dataset(
    test_array,
    input_sequence_length,
    forecast_horizon,
    batch_size=test_array.shape[0],
    shuffle=False,
    multi_horizon=multi_horizon,
)


node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
graph = GraphInfo(
    edges=(node_indices.tolist(), neighbor_indices.tolist()),
    num_nodes=adjacency_matrix.shape[0],
)
#print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")


graph_conv_params = {
    "aggregation_type": "mean",
    "combination_type": "concat",
}

st_gcn = CFCG(
    in_feat,
    out_feat,
    lstm_units,
    input_sequence_length,
    forecast_horizon,
    graph,
    graph_conv_params,
)
inputs = layers.Input((input_sequence_length, graph.num_nodes, in_feat))
outputs = st_gcn(inputs)


model = keras.models.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0002),
    loss=keras.losses.MeanSquaredError(),
)
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[keras.callbacks.EarlyStopping(patience=10)],
)


x_test, y = next(test_dataset.as_numpy_iterator())
y_pred = model.predict(x_test)


x_test = x_test*std + mean
y_pred = y_pred*std[:,0] + mean[:,0]
y = y*std[:,0] + mean[:,0]


totalmodel_mape = (np.abs((y_pred[:, 0:24, :] - y[:, 0:24, :])/y[:, 0:24, :])).mean()
mape_country = np.average(np.abs((y_pred[:, 0:24, :] - y[:, 0:24, :])/y[:, 0:24, :]), axis=0) 
mape_country = np.average(mape_country,axis=0)
print(f" total model MAPE: {totalmodel_mape}")


