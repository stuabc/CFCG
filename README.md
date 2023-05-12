# CFCG
A GNN-based Day Ahead Carbon Intensity Forecasting Model for Cross-Border Power Grids  

1. Data sources  
The European electricity data is published by [ENTSOE](https://transparency.entsoe.eu/dashboard/show?loggedUserIsPrivileged=false). CFCG need the following data:  
(1)Hourly history carbon intensity (electricity generation by source)    
(2)Hourly history electricity flows  


2. Environmental dependence  
Keras & Tensorflow 2, Python 3, numpy, pandas, matplotlib  

3. Usage  
3.1  Calculating average local (generation-based) history carbon intensity:  
For calculating history local (generation-based) carbon intensity , run the following file: Local_CI_calculation.py  
Note: we use emission factor method to calculate the history carbon intensity in a local power grid.  
3.2 Calculating average corss-border carbon intensity:  
For calculating cross-border history carbon intensity involved electricy exchange using direct couple scheme, run the following file: Crossborder_CI_account_direct.py  
For calculating cross-border history carbon intensity involved electricy exchange using aggreated couple scheme, run the following file: Crossborder_CI_account_agg.py  
Note: In a cross-border power grid, the carbon intensity is affected by electricity exchanges. For details about calculating schemes, you can see the follwoing references : 
[1] [A Quasi-Input-Output model to improve the estimation of emission factors for purchased electricity from interconnected grids](https://doi.org/10.1016/j.apenergy.2017.05.046)
[2] [Real-time carbon accounting method for the European electricity markets](https://doi.org/10.1016/j.esr.2019.100367)
[3] [Tracing carbon dioxide emissions in the European electricity markets](https://ieeexplore.ieee.org/document/9221928)  
3.3 Forecast the carbon intensity in Cross-Border Power Grids:  
For forecasting cross-border carbon intensity, run the following file: cfcg-main.py

