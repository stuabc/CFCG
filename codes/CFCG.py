import pandas as pd
import numpy as np
import os
import typing
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

from keras.models import load_model


train_size, val_size = 0.6, 0.2
batch_size = 64
input_sequence_length = 24
forecast_horizon = 24
multi_horizon = True

country = ['Austria (AT)', 'Belgium (BE)','Bulgaria (BG)','Switzerland (CH)','Cyprus (CY)', 
          'Czech Republic (CZ)','Germany (DE)','Denmark (DK)','Estonia (EE)','Spain (ES)','Finland (FI)','France (FR)',
          'Greece (GR)','Croatia (HR)','Hungary (HU)','Ireland (IE)','Italy (IT)','Lithuania (LT)','Latvia (LV)',
          'Netherlands (NL)','Norway (NO)','Poland (PL)','Portugal (PT)','Serbia (RS)','Romania (RO)',
          'Sweden (SE)','Slovenia (SI)','Slovakia (SK)']


def preprocess(data_array: np.ndarray, train_size: float, val_size: float):

    
    num_time_steps = data_array.shape[0]
    num_train, num_val = (
        int(num_time_steps * train_size),
        int(num_time_steps * val_size),
    )
    train_array = data_array[:num_train]
    mean, std = train_array.mean(axis=0), train_array.std(axis=0)

    train_array = (train_array - mean) / std
    val_array = (data_array[num_train : (num_train + num_val)] - mean) / std
    test_array = (data_array[(num_train + num_val) :] - mean) / std
    train_array[np.isnan(train_array)] = 0
    val_array[np.isnan(val_array)] = 0
    test_array[np.isnan(test_array)] = 0
    train_array[np.isinf(train_array)] = 0
    val_array[np.isinf(val_array)] = 0
    test_array[np.isinf(test_array)] = 0


    return train_array, val_array, test_array,mean,std

def create_tf_dataset(
    data_array: np.ndarray,
    input_sequence_length: int,
    forecast_horizon: int,
    batch_size: int = 128,
    shuffle=True,
    multi_horizon=True,
):
    """Creates tensorflow dataset from numpy array.

    """

    inputs = timeseries_dataset_from_array(
        #np.expand_dims(data_array[:-forecast_horizon], axis=-1),
        data_array[:-forecast_horizon],
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
        
    )

    target_offset = (
        input_sequence_length
        if multi_horizon
        else input_sequence_length + forecast_horizon - 1
    )
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = timeseries_dataset_from_array(
        data_array[target_offset:][:,:,0], 
        None,
        sequence_length=target_seq_length,
        shuffle=False,
        batch_size=batch_size,
       
    )

    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)

    return dataset.prefetch(16).cache()


def get_all_csv_name(path):
    filename_list = []
    for folderName, subfolders, filenames in os.walk(path):
        for file_name in filenames:
            if '.csv' in file_name:
                filename_list.append(file_name)
    return filename_list

###calcutation transfer
# pathtrans = "C:/poster/FLOW"
# pathgen = "C:/poster/GEN"

# filename_gen = get_all_csv_name(pathgen)
# filename_trans = get_all_csv_name(pathtrans)

# num_nodes = 28
# time_steps_hours = 26304 #3 years

# #transfer = np.zeros((num_nodes,num_nodes,time_steps_hours)) 
# transfer = np.zeros((time_steps_hours,num_nodes,num_nodes)) 
# generation = np.zeros((time_steps_hours,num_nodes))
# elecinflow = np.zeros((time_steps_hours,num_nodes))
# total_power = np.zeros((time_steps_hours,num_nodes))
# ex = np.zeros((time_steps_hours,num_nodes))
# B = np.zeros((num_nodes,num_nodes,time_steps_hours))
# ci_direct = np.zeros((time_steps_hours,num_nodes)) 
# ci_network = np.zeros((time_steps_hours,num_nodes)) 
# outcarbonflow = np.zeros((time_steps_hours,num_nodes))
# incarbonflow = np.zeros((time_steps_hours,num_nodes))

# ci_net = pd.read_csv("C:/poster/ci_network.csv")
# # ci_net = pd.read_csv("C:/poster/ci_network.csv")
# #ci_net = pd.read_csv("C:/poster/ci_direct.csv")
# # ci_net = pd.read_csv("C:/poster/ci_network_newaccount.csv")
# ci_ground = pd.read_csv("C:/poster/ci_direct.csv")
# speedsci = ci_net[ci_net.columns[1:]].values


# for i in filename_trans:
#     dataset_trans = pd.read_csv(pathtrans +'/' + i, header=0, low_memory=False)
#     dataset_trans.replace('n/e',0,inplace=True)
#     dataset_trans.replace('#VALUE!',None,inplace=True)
#     dataset_trans.fillna(method = 'ffill', inplace=True)
#     dataset_trans.fillna(method = 'bfill', inplace=True)
#     indexsource = country.index(list(dataset_trans)[1][0:list(dataset_trans)[1].index('>')-1])
#     indextarget = country.index(list(dataset_trans)[1][list(dataset_trans)[1].index('>')+2:len(list(dataset_trans)[1])-5])
#     transfer[:,indexsource,indextarget] = dataset_trans.iloc[:,1]
#     indexsource = country.index(list(dataset_trans)[2][0:list(dataset_trans)[2].index('>')-1])
#     indextarget = country.index(list(dataset_trans)[2][list(dataset_trans)[2].index('>')+2:len(list(dataset_trans)[2])-5])
#     transfer[:,indexsource,indextarget] = dataset_trans.iloc[:,2]



# for i in filename_gen:
#     dataset_gen = pd.read_csv(pathgen +'/'+ i, header=0, low_memory=False)
#     dataset_gen.replace('n/e',None,inplace=True)
#     dataset_gen.replace('#VALUE!',None,inplace=True)
#     dataset_gen.fillna(method = 'ffill', inplace=True)
#     dataset_gen.fillna(method = 'bfill', inplace=True)
#     index = country.index(dataset_gen['Area'][0])
#     generation[:,index] = dataset_gen['Total']
#     ci_direct[:,index] = dataset_gen['CI_lce']
    
    
# total_power = generation
# for i in range(num_nodes):
#     for j in range(time_steps_hours):
#         elecinflow[j,i] = sum(transfer[j,:,i])
#         total_power[j,i] = total_power[j,i] + sum(transfer[j,:,i])
#         # total_power[j,i] = total_power[j,i] - sum(transfer[i,:,j])
#         # if total_power[j,i] <=0:
#         #     total_power[j,i] = total_power[j-1,i]

# for i in range(num_nodes):
#     for j in range(num_nodes):
#         for t in range(time_steps_hours):
# #            outcarbonflow[t,i] = outcarbonflow[t,i]+transfer[i,j,t] * ci_arrays[t,i] # out carbon flow
#             incarbonflow[t,i] = incarbonflow[t,i]+transfer[t,j,i] * speedsci[t,j] # in carbon flow   



# ep = generation*ci_direct  # total emission / kg
# #outcarbonratio = outcarbonflow/total_power
# incarbonratio = incarbonflow/total_power
###计end transfer


transfer= np.load('d:/transfer.npy')
ci_net = pd.read_csv("D:/ci_network.csv")
# ci_net = pd.read_csv("C:/poster/ci_network.csv")
#ci_net = pd.read_csv("C:/poster/ci_direct.csv")
# ci_net = pd.read_csv("C:/poster/ci_network_newaccount.csv")
#ci_ground = pd.read_csv("C:/poster/ci_direct.csv")
ci_arrays = ci_net[ci_net.columns[1:]].values
#speeds_ground = ci_ground[ci_ground.columns[1:]].values


# ci_arrays = np.stack((ci_arrays[24*7:],
#                             ci_arrays[24*7:]) ,axis=2)


ci_arrays = np.stack((ci_arrays[24*7:-24], ci_arrays[24*6:-24*2], ci_arrays[:-24*8],
                          ) ,axis=2)

#transfer_mean = np.mean(transfer,0)


# ci_arrays = np.stack((ci_arrays[24*7:-24], ci_arrays[24*6:-24*2], ci_arrays[:-24*8],
#                           transfer[24*7:-24,0,:],transfer[24*7:-24,1,:],transfer[24*7:-24,2,:],
#                           transfer[24*7:-24,3,:],transfer[24*7:-24,4,:],transfer[24*7:-24,5,:],
#                           transfer[24*7:-24,6,:],transfer[24*7:-24,7,:],transfer[24*7:-24,8,:],
#                           transfer[24*7:-24,9,:],transfer[24*7:-24,10,:],transfer[24*7:-24,11,:],
#                           transfer[24*7:-24,12,:],transfer[24*7:-24,13,:],transfer[24*7:-24,14,:],
#                           transfer[24*7:-24,15,:],transfer[24*7:-24,16,:],transfer[24*7:-24,17,:],
#                           transfer[24*7:-24,18,:],transfer[24*7:-24,19,:],transfer[24*7:-24,20,:],
#                           transfer[24*7:-24,21,:],transfer[24*7:-24,22,:],transfer[24*7:-24,23,:],
#                           transfer[24*7:-24,24,:],transfer[24*7:-24,25,:],transfer[24*7:-24,26,:],
#                           transfer[24*7:-24,27,:]) ,axis=2)


# carbonchangerete = ( np.mean(speeds_ground[:-24*360,:],axis=0) - np.mean(ci_arrays[:-24*360,:],axis=0) ) / np.mean(ci_arrays[:-24*360,:],axis=0)
# carbonchangerete = pd.DataFrame(carbonchangerete,index=country)
# carbonchangerete.to_csv("carbonchangerate_2021.csv",sep=',',header=True) 



# cacc_mae = np.mean(abs (speeds_ground[24:,] - speeds_ground[:-24,] )/speeds_ground[24:,])



train_array, val_array, test_array,mean,std = preprocess(ci_arrays, train_size, val_size)
print(f"train set size: {train_array.shape}")
print(f"validation set size: {val_array.shape}")
print(f"test set size: {test_array.shape}")

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


class GraphInfo:
    def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
        self.edges = edges
        self.num_nodes = num_nodes



adjacency_matrix = pd.read_csv("d:/Adj.csv",header=None).values
node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
graph = GraphInfo(
    edges=(node_indices.tolist(), neighbor_indices.tolist()),
    num_nodes=adjacency_matrix.shape[0],
)
print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")


class GraphConv(layers.Layer):
    def __init__(
        self,
        in_feat,
        out_feat,
        graph_info: GraphInfo,
        aggregation_type="mean",
        combination_type="concat",
        activation: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.weight = tf.Variable(
            initial_value=keras.initializers.glorot_uniform()(
                shape=(in_feat, out_feat), dtype="float32"
            ),
            trainable=True,
        )
        self.activation = layers.Activation(activation)

    def aggregate(self, neighbour_representations: tf.Tensor):
        aggregation_func = {
            "sum": tf.math.unsorted_segment_sum,
            "mean": tf.math.unsorted_segment_mean,
            "max": tf.math.unsorted_segment_max,
        }.get(self.aggregation_type)

        if aggregation_func:
            return aggregation_func(
                neighbour_representations,
                self.graph_info.edges[0],
                num_segments=self.graph_info.num_nodes,
            )

        raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

    def compute_nodes_representation(self, features: tf.Tensor):
        """Computes each node's representation.
        """
        return tf.matmul(features, self.weight)

    def compute_aggregated_messages(self, features: tf.Tensor):
        neighbour_representations = tf.gather(features, self.graph_info.edges[1])
        aggregated_messages = self.aggregate(neighbour_representations)
        return tf.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation: tf.Tensor, aggregated_messages: tf.Tensor):
        if self.combination_type == "concat":
            h = tf.concat([nodes_representation, aggregated_messages], axis=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        return self.activation(h)

    def call(self, features: tf.Tensor):
        """Forward pass.
        """
        nodes_representation = self.compute_nodes_representation(features)
        aggregated_messages = self.compute_aggregated_messages(features)
        return self.update(nodes_representation, aggregated_messages)
    
    
    
class LSTMGC(layers.Layer):
    """Layer comprising a convolution layer followed by LSTM and dense layers."""

    def __init__(
        self,
        in_feat,
        out_feat,
        lstm_units: int,
        input_seq_len: int,
        output_seq_len: int,
        graph_info: GraphInfo,
        graph_conv_params: typing.Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # graph conv layer
        if graph_conv_params is None:
            graph_conv_params = {
                "aggregation_type": "mean",
                "combination_type": "concat",
                "activation": None,
            }
        self.graph_conv = GraphConv(in_feat, out_feat, graph_info, **graph_conv_params)

        self.lstm = layers.LSTM(lstm_units, activation="relu")
        self.dense = layers.Dense(output_seq_len)

        self.input_seq_len, self.output_seq_len = input_seq_len, output_seq_len

    def call(self, inputs):
        """Forward pass.

        """

        # convert shape to  (num_nodes, batch_size, input_seq_len, in_feat)
       
        inputs = tf.transpose(inputs, [2, 0, 1, 3])
        # print(tf.shape(inputs),'before')
        # inputs = inputs[:,:,:,0:1]
        # print(tf.shape(inputs),'after')

        gcn_out = self.graph_conv(
            inputs
        )  # gcn_out has shape: (num_nodes, batch_size, input_seq_len, out_feat)
        shape = tf.shape(gcn_out)
        num_nodes, batch_size, input_seq_len, out_feat = (
            shape[0],
            shape[1],
            shape[2],
            shape[3],
        )

        # LSTM takes only 3D tensors as input
        gcn_out = tf.reshape(gcn_out, (batch_size * num_nodes, input_seq_len, out_feat))
        lstm_out = self.lstm(
            gcn_out
        )  # lstm_out has shape: (batch_size * num_nodes, lstm_units)

        dense_output = self.dense(
            lstm_out
        )  # dense_output has shape: (batch_size * num_nodes, output_seq_len)
        output = tf.reshape(dense_output, (num_nodes, batch_size, self.output_seq_len))
        return tf.transpose(
            output, [1, 2, 0]
        )  # returns Tensor of shape (batch_size, output_seq_len, num_nodes)
    
    
in_feat = 3
#batch_size = 64
epochs = 2



#input_sequence_length = 24
#forecast_horizon = 24
#multi_horizon = True
out_feat = 3
lstm_units = 32
graph_conv_params = {
    "aggregation_type": "mean",
    "combination_type": "concat",
    "activation": None,
}

st_gcn = LSTMGC(
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



#model = load_model('GNN')
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

#model.save('GNN5epoch')


#model = load_model('GNN')

x_test, y = next(test_dataset.as_numpy_iterator())
y_pred = model.predict(x_test)



# x_test = x_test*std + mean
# y_pred = y_pred*std + mean
# y = y*std + mean

x_test = x_test*std + mean
y_pred = y_pred*std[:,0] + mean[:,0]
y = y*std[:,0] + mean[:,0]



df_ci_gnn_truth = pd.DataFrame(y[:,0,:],columns=country)
#df_ci_gnn_truth.to_csv("ci_gnn_truth.csv",sep=',',header=True) 

df_ci_gnn_predict = pd.DataFrame(y_pred[:,0,:],columns=country)
#df_ci_gnn_predict.to_csv("ci_gnn_predict.csv",sep=',',header=True) 



totalmodel_mape = (np.abs((y_pred[:, 0:24, :] - y[:, 0:24, :])/y[:, 0:24, :])).mean()
mape_country = np.average(np.abs((y_pred[:, 0:24, :] - y[:, 0:24, :])/y[:, 0:24, :]), axis=0) 

mape_country = np.average(mape_country,axis=0)
print(f" total model MAPE: {totalmodel_mape}")

# dataframe = pd.DataFrame(y_pred[:, 0, :],columns=country) 
# dataframe.to_csv("C:/poster/final/predictionvalue/forecast_CFCG_best_13.5_single.csv", index=False, sep=',')
# dataframe = pd.DataFrame(y[:, 0, :],columns=country)
# dataframe.to_csv("C:/poster/final/predictionvalue/truth_CFCG_best_13.5_single.csv", index=False, sep=',')


# df_mae_direct = pd.DataFrame(mae_country,index=country)
# df_rmse_direct = pd.DataFrame(rmse_country,index=country)
# df_mape_direct = pd.DataFrame(mape_country,index=country)
# df_mae_direct.to_csv("c:/poster/final/metric/CFCG_mae_best_13.5.csv",sep=',')
# df_rmse_direct.to_csv("c:/poster/final/metric/CFCG_rmse_best_13.5.csv",sep=',')
# df_mape_direct.to_csv("c:/poster/final/metric/CFCG_mape_best_13.5.csv",sep=',')

