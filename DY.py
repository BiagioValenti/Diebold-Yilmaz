#THIS CODE PERFORMS DIEBOLD-YILMAZ ANALYISIS OF VOLATILITY OF A STOCK MARKET
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.tsatools import lagmat
import networkx as nx
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.optim as optim

#Define ADF test function
def adf_test(series, name="Serie"):
    result = adfuller(series, autolag="AIC")
    print(f"--- ADF Test: {name} ---")
    print(f"ADF Statistic : {result[0]:.4f}")
    print(f"p-value       : {result[1]:.20f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key} : {value:.4f}")
    
    if result[1] < 0.05:
        print("→ Serie stazionaria (rigetto H0)")
    else:
        print("→ Serie non stazionaria (non rigetto H0)")
    print("\n")


#Folder
folder = r"C:\Users\bival\Documents\Dottorato\Corsi\Econometrics II\Indici" 

#File opening 
dataframes = {}
for filename in os.listdir(folder):
    if filename.endswith(".csv"):

        path = os.path.join(folder, filename)
        df = pd.read_csv(path)

        df["Date"] = pd.to_datetime(df["Date"])

        df["Price"] = (
            df["Price"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace(["", "null", "NaN", "-", "None"], np.nan)
        )

        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

        df = df.sort_values("Date")

        code = filename.replace(".csv", "")
        dataframes[code] = df


#Start ate alignement
start_dates = []

for code, df in dataframes.items():
    start_dates.append(df["Date"].min())

common_start_date = max(start_dates)

print("Common start date:", common_start_date.date())

for code in dataframes:
    dataframes[code] = dataframes[code][
        dataframes[code]["Date"] >= common_start_date
    ].reset_index(drop=True)


#Prices plot
plt.figure(figsize=(12, 6))

for code, df in dataframes.items():
    plt.plot(df["Date"], df["Price"], label=code)

plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Prices evolution over time")
plt.legend()
plt.grid(True)

plt.tight_layout()
#plt.show()
plt.savefig(f"Prices.png")

#Calculate returns
returns = {}

for code, df in dataframes.items():
    df = df.copy()

    # evita log(0)
    df = df[df["Price"] > 0]

    df["log_price"] = np.log(df["Price"])
    df["return"] = df["log_price"].diff()

    df = df.dropna(subset=["return"])

    returns[code] = df[["Date", "return"]]

#Calculate and plot volatility
volatility = {}

for code, df in returns.items():
    df = df.copy()

    df["volatility"] = df["return"] ** 2

    volatility[code] = df[["Date", "volatility"]]

plt.figure(figsize=(12, 6))
for code, df in volatility.items():
    plt.plot(df["Date"], df["volatility"], label=code)

plt.title("Sectoral volatility (squared returns)")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.grid(True)
plt.savefig(f"volatility.png")
#plt.show()

#ADF tests
print("ADF test returns")
for code, df in returns.items():
    adf_test(df["return"], name=code)

print("ADF test volatility")
for code, df in volatility.items():
    adf_test(df["volatility"], name=code)

#Start of DY approach
use_data = volatility  

# ICB sectors list
sectors = list(use_data.keys())

#Use aligned dates
common_dates = set(use_data[sectors[0]]["Date"])
for sec in sectors[1:]:
    common_dates = common_dates.intersection(set(use_data[sec]["Date"]))
common_dates = sorted(list(common_dates))

#VAR
var_df = pd.DataFrame({"Date": common_dates})

for sec in sectors:
    df = use_data[sec].set_index("Date")
    data_col = [col for col in df.columns if col != "Date"][0]
    var_df[sec] = df.loc[common_dates, data_col].values

var_df = var_df.reset_index(drop=True)
#print(var_df.head())


#VAR lag order with AIC
model = VAR(var_df[sectors])
lag_order = model.select_order(maxlags=20).aic
print("Selected lag (AIC):", lag_order)

# Initial VAR for initial coefficients
var_results = model.fit(lag_order)
#print(var_results.summary())

# Elastic Net
Y = var_df[sectors].values
X_lagged = lagmat(Y, maxlag=lag_order, trim='both')
Y_trunc = Y[lag_order:]
X_lagged = np.hstack([np.ones((len(X_lagged), 1)), X_lagged])  

coef_init = var_results.params.values.T  

# 80/20 split
split_idx = int(0.8 * len(Y_trunc))
X_train, X_valid = X_lagged[:split_idx], X_lagged[split_idx:]
Y_train, Y_valid = Y_trunc[:split_idx], Y_trunc[split_idx:]

class ElasticNetVAR(nn.Module):
    def __init__(self, coef_init):
        super().__init__()
        self.coef = nn.Parameter(torch.tensor(coef_init, dtype=torch.float32))
    
    def forward(self, X):
        return X @ self.coef.T 

# Tuning grid
lambdas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  

best_loss = float('inf')
best_l1_lambda = None
best_l2_lambda = None
best_coef = None

for alpha in alphas:
    for lambda_val in lambdas:
        l1_lambda = alpha * lambda_val
        l2_lambda = (1 - alpha) * lambda_val
        
        #Model
        model_en = ElasticNetVAR(coef_init)
        optimizer = optim.Adam(model_en.parameters(), lr=0.01)
        
        # Train on train set
        X_t_train = torch.tensor(X_train, dtype=torch.float32)
        Y_t_train = torch.tensor(Y_train, dtype=torch.float32)
        for epoch in range(200):
            pred_train = model_en(X_t_train)
            mse_loss = nn.MSELoss()(pred_train, Y_t_train)
            l1_loss = l1_lambda * torch.sum(torch.abs(model_en.coef))
            l2_loss = l2_lambda * torch.sum(model_en.coef ** 2)
            loss = mse_loss + l1_loss + l2_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Vlidation on valid set
        X_t_valid = torch.tensor(X_valid, dtype=torch.float32)
        Y_t_valid = torch.tensor(Y_valid, dtype=torch.float32)
        with torch.no_grad():
            pred_valid = model_en(X_t_valid)
            valid_loss = nn.MSELoss()(pred_valid, Y_t_valid).item()
        
        print(f"Lambda: {lambda_val}, Alpha: {alpha}, Valid Loss: {valid_loss}")
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_l1_lambda = l1_lambda
            best_l2_lambda = l2_lambda
            best_coef = model_en.coef.detach().numpy()

print(f"Best params: l1_lambda={best_l1_lambda}, l2_lambda={best_l2_lambda}, Valid Loss={best_loss}")

model_en_full = ElasticNetVAR(coef_init)
optimizer_full = optim.Adam(model_en_full.parameters(), lr=0.01)

X_t_full = torch.tensor(X_lagged, dtype=torch.float32)
Y_t_full = torch.tensor(Y_trunc, dtype=torch.float32)
for epoch in range(200):
    pred_full = model_en_full(X_t_full)
    mse_loss = nn.MSELoss()(pred_full, Y_t_full)
    l1_loss = best_l1_lambda * torch.sum(torch.abs(model_en_full.coef))
    l2_loss = best_l2_lambda * torch.sum(model_en_full.coef ** 2)
    loss = mse_loss + l1_loss + l2_loss
    optimizer_full.zero_grad()
    loss.backward()
    optimizer_full.step()

shrunk_coef = model_en_full.coef.detach().numpy()
print("Number of final shrinked coefficients:", shrunk_coef.round(3))
print("Numero of coefficients ~zero:", np.sum(np.abs(shrunk_coef) < 1e-3))

resid = Y_trunc - X_lagged @ shrunk_coef.T
sigma_u = np.cov(resid.T)

#horizon choice
horizon = 10  

#Generalised fevd
def generalized_fevd_custom(shrunk_coef, sigma_u, horizon, k, lag_order):
    const = shrunk_coef[:, 0]
    A_comp = shrunk_coef[:, 1:].reshape(k, k, lag_order)  # (k, k, p)

    #Companion matrix for IRFs 
    companion = np.zeros((k*lag_order, k*lag_order))
    for i in range(lag_order):
        companion[:k, i*k:(i+1)*k] = A_comp[:, :, i].T
    if lag_order > 1:
        companion[k:, :-k] = np.eye(k*(lag_order-1))

    #IRFs
    irfs = np.zeros((horizon, k, k))
    irfs[0] = np.eye(k) 
    comp_pow = np.eye(k*lag_order)
    for h in range(1, horizon):
        comp_pow = comp_pow @ companion
        irfs[h] = comp_pow[:k, :k]

    num = np.zeros((k, k))
    den = np.zeros(k)
    for h in range(horizon):
        Phi_h = irfs[h]
        for i in range(k):
            den[i] += Phi_h[i, :] @ sigma_u @ Phi_h[i, :].T
            for j in range(k):
                temp = Phi_h[i, :] @ sigma_u[:, j]
                num[i, j] += (temp ** 2) / sigma_u[j, j]
    
    fevd = np.zeros((k, k))
    for i in range(k):
        fevd[i, :] = num[i, :] / den[i] if den[i] != 0 else 0
    
    # Normalise for row
    for i in range(k):
        row_sum = np.sum(fevd[i, :])
        if row_sum > 0:
            fevd[i, :] /= row_sum  
    return fevd


k = len(sectors)
fevd_mean = generalized_fevd_custom(shrunk_coef, sigma_u, horizon, k, lag_order)
print(fevd_mean)
fevd_mean *= 100
print(f"Row sums = {np.sum(fevd_mean, axis=1)}")  #Normalisation check
print("Calculated GFEVD:", fevd_mean.shape)  # (k, k)

#Diagonal
diag = np.diag(fevd_mean)

# Total Spillover Index
total_spillover = 100 * (np.sum(fevd_mean) - np.sum(diag)) / np.sum(fevd_mean)
print("Total Spillover Index:", total_spillover)


# Directional TO others
directional_to = (fevd_mean.sum(axis=0) - np.diag(fevd_mean))
# Directional FROM others
directional_from = (fevd_mean.sum(axis=1) - np.diag(fevd_mean))
# Net
net_spillover = directional_to - directional_from

print("Directional To:", directional_to)
print("Directional From:", directional_from)
print("Net Spillover:", net_spillover)

#Dynamic estimation

#Window size and steps
window = 200  
steps = 10
total_spillover_series = []
directional_to_series = [[] for _ in range(k)]
directional_from_series = [[] for _ in range(k)]
net_spillover_series = [[] for _ in range(k)]

#Ending dates
dates_for_plot = var_df['Date'].iloc[window::steps]

#Rolling window calculation
for start in range(0, len(var_df) - window, steps):
    try:
        var_sub = var_df[sectors].iloc[start:start+window]
        current_date = var_df['Date'].iloc[start + window - 1]

        model_sub = VAR(var_sub)
        var_results_sub = model_sub.fit(lag_order)

        # Elastic Net
        Y_sub = var_sub.values
        X_lagged_sub = lagmat(Y_sub, maxlag=lag_order, trim='both')
        Y_trunc_sub = Y_sub[lag_order:]
        X_lagged_sub = np.hstack([np.ones((len(X_lagged_sub), 1)), X_lagged_sub])

        coef_init_sub = var_results_sub.params.values.T

        model_en_sub = ElasticNetVAR(coef_init_sub)
        optimizer_sub = optim.Adam(model_en_sub.parameters(), lr=0.01)

        X_t_sub = torch.tensor(X_lagged_sub, dtype=torch.float32)
        Y_t_sub = torch.tensor(Y_trunc_sub, dtype=torch.float32)

        for epoch in range(200):
            pred_sub = model_en_sub(X_t_sub)
            mse_loss = nn.MSELoss()(pred_sub, Y_t_sub)
            l1_loss = best_l1_lambda * torch.sum(torch.abs(model_en_sub.coef))
            l2_loss = best_l2_lambda * torch.sum(model_en_sub.coef ** 2)
            loss = mse_loss + l1_loss + l2_loss
            optimizer_sub.zero_grad()
            loss.backward()
            optimizer_sub.step()

        shrunk_coef_sub = model_en_sub.coef.detach().numpy()

        resid_sub = Y_trunc_sub - X_lagged_sub @ shrunk_coef_sub.T
        sigma_u_sub = np.cov(resid_sub.T)

        # GFEVD
        fevd_sub = generalized_fevd_custom(shrunk_coef_sub, sigma_u_sub, horizon, k, lag_order)
        fevd_sub *= 100

        #TSI
        diag_sub = np.diag(fevd_sub)
        TSI = (np.sum(fevd_sub) - np.sum(diag_sub)) / np.sum(fevd_sub) * 100
        directional_from_sub = np.sum(fevd_sub, axis=0) - diag_sub
        directional_to_sub = np.sum(fevd_sub, axis=1) - diag_sub
        net_spillover_sub = directional_to_sub - directional_from_sub

        #Append
        total_spillover_series.append(TSI)
        for j in range(k):
            directional_to_series[j].append(directional_to_sub[j])
            directional_from_series[j].append(directional_from_sub[j])
            net_spillover_series[j].append(net_spillover_sub[j])

    except Exception as e:
        print(f"Error {start}: {e}")
        total_spillover_series.append(np.nan)
        for j in range(k):
            directional_to_series[j].append(np.nan)
            directional_from_series[j].append(np.nan)
            net_spillover_series[j].append(np.nan)


sectors = var_df.columns[1:]

#Network function (plot and analysis)
def plot_spillover_network(
    fevd_mean, 
    sectors, 
    weight_threshold=0.0,   # only link > treshold
    node_size=1000,         
    arrow_size=40,
    arrow_distance=0.1,     
    cmap=plt.cm.viridis
):
    #Graph object
    G = nx.DiGraph()
    
    for sec in sectors:
        G.add_node(sec)
    
    #Edges with weight filter
    for i, src in enumerate(sectors):
        for j, tgt in enumerate(sectors):
            if i != j and fevd_mean[i,j] > weight_threshold:
                G.add_edge(tgt, src, weight=fevd_mean[i,j])

    nx.write_gml(G, "spillover_network.gml") 
    
    pos = nx.circular_layout(G, scale=3)
    
    fig, ax = plt.subplots(figsize=(10,10))
    
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='skyblue', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)

    #Arrows
    edges = G.edges(data=True)
    weights = [1 for (_,_,d) in edges] 
    edge_colors = [d['weight'] for (_,_,d) in edges]
    
    #Colors
    max_weight = max(edge_colors) if edge_colors else 1
    min_weight = min(edge_colors) if edge_colors else 0
    if max_weight == min_weight:
        edge_colors_norm = [0.5 for w in edge_colors]
    else:
        edge_colors_norm = [(w - min_weight)/(max_weight - min_weight) for w in edge_colors]
    
    nx.draw_networkx_edges(
        G, pos, edgelist=G.edges(), width=weights,
        arrowstyle='-|>', arrowsize=arrow_size, edge_color=edge_colors_norm, edge_cmap=cmap,
        ax=ax, connectionstyle=f'arc3,rad={arrow_distance}'
    )
    
    # Colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=min_weight, vmax=max_weight))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Spillover (%)", rotation=270, labelpad=15)
    
    ax.set_title("Directional Spillovers Network")
    ax.axis('off')
    #plt.show()
    plt.savefig(f"Network_{weight_threshold}.png")

    # Measures
    in_degree = dict(G.in_degree(weight='weight'))  # Weighted in-degree 
    out_degree = dict(G.out_degree(weight='weight'))  # Weighted out-degree 


    closeness = nx.closeness_centrality(G, distance=lambda u,v,d: 1/d['weight'])
    eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=1000) 
    print("Closeness Centrality:", sorted(closeness.items(), key=lambda x: x[1], reverse=True))
    print("Eigenvector Centrality:", sorted(eigenvector.items(), key=lambda x: x[1], reverse=True))

    # Clustering
    clustering = nx.clustering(nx.Graph(G), weight='weight') 
    print("Clustering Coefficient (local):", clustering)

    # Assortativity
    assort_degree = nx.degree_assortativity_coefficient(G, weight='weight')
    print("Degree Assortativity:", assort_degree) 


plot_spillover_network(
    fevd_mean, sectors, 
    weight_threshold=0, 
    node_size=800, 
    arrow_size=40,
    arrow_distance=0.2
)


plot_spillover_network(
    fevd_mean, sectors, 
    weight_threshold=10, 
    node_size=800, 
    arrow_size=40,
    arrow_distance=0.2
)


# Robustness checks

# 1. Different lag order (use BIC, refit VAR/FEVD/TSI)
lag_bic = model.select_order(maxlags=20).bic
print(f"Lag BIC={lag_bic} (vs AIC={lag_order})")

#BIC
var_results_bic = model.fit(lag_bic)
coef_init_bic = var_results_bic.params.values.T

X_lagged_bic = lagmat(Y, maxlag=lag_bic, trim='both')
Y_trunc_bic = Y[lag_bic:]
X_lagged_bic = np.hstack([np.ones((len(X_lagged_bic), 1)), X_lagged_bic])

#Elastic Net
model_en_bic = ElasticNetVAR(coef_init_bic)
optimizer_bic = optim.Adam(model_en_bic.parameters(), lr=0.01)

X_t_bic = torch.tensor(X_lagged_bic, dtype=torch.float32)
Y_t_bic = torch.tensor(Y_trunc_bic, dtype=torch.float32)
for epoch in range(200):
    pred_bic = model_en_bic(X_t_bic)
    mse_loss = nn.MSELoss()(pred_bic, Y_t_bic)
    l1_loss = best_l1_lambda * torch.sum(torch.abs(model_en_bic.coef))
    l2_loss = best_l2_lambda * torch.sum(model_en_bic.coef ** 2)
    loss = mse_loss + l1_loss + l2_loss
    optimizer_bic.zero_grad()
    loss.backward()
    optimizer_bic.step()

shrunk_coef_bic = model_en_bic.coef.detach().numpy()

resid_bic = Y_trunc_bic - X_lagged_bic @ shrunk_coef_bic.T
sigma_u_bic = np.cov(resid_bic.T)

#GFEVD
fevd_bic = generalized_fevd_custom(shrunk_coef_bic, sigma_u_bic, horizon, k, lag_bic) * 100
diag_bic = np.diag(fevd_bic)
tsi_bic = (np.sum(fevd_bic) - np.sum(diag_bic)) / np.sum(fevd_bic) * 100
print(f"TSI con Lag BIC: {tsi_bic:.2f}%")

# 2. Different horizon (refit FEVD/TSI)
horizons = [5, 10, 15]
for h in horizons:
    fevd_rob = generalized_fevd_custom(shrunk_coef, sigma_u, h, k, lag_order) * 100
    print(f"Horizon={h}: Row sums = {np.sum(fevd_rob, axis=1)}") 
    diag_rob = np.diag(fevd_rob)
    tsi_rob = (np.sum(fevd_rob) - np.sum(diag_rob)) / np.sum(fevd_rob) * 100
    print(f"Horizon={h}: TSI={tsi_rob:.6f}%")

# 3. Different window size 
windows = [150, 200, 250, 300, 350]
for w in windows:
    sub_series = []
    for start in range(0, len(var_df) - w, 10): 
        try:
            var_sub = var_df[sectors].iloc[start:start+w]

            model_sub = VAR(var_sub)
            var_results_sub = model_sub.fit(lag_order)
            
            Y_sub = var_sub.values
            X_lagged_sub = lagmat(Y_sub, maxlag=lag_order, trim='both')
            Y_trunc_sub = Y_sub[lag_order:]
            X_lagged_sub = np.hstack([np.ones((len(X_lagged_sub), 1)), X_lagged_sub])
            
            coef_init_sub = var_results_sub.params.values.T
            
            model_en_sub = ElasticNetVAR(coef_init_sub)
            optimizer_sub = optim.Adam(model_en_sub.parameters(), lr=0.01)
            
            X_t_sub = torch.tensor(X_lagged_sub, dtype=torch.float32)
            Y_t_sub = torch.tensor(Y_trunc_sub, dtype=torch.float32)
            
            for epoch in range(200):
                pred_sub = model_en_sub(X_t_sub)
                mse_loss = nn.MSELoss()(pred_sub, Y_t_sub)
                l1_loss = best_l1_lambda * torch.sum(torch.abs(model_en_sub.coef))
                l2_loss = best_l2_lambda * torch.sum(model_en_sub.coef ** 2)
                loss = mse_loss + l1_loss + l2_loss
                optimizer_sub.zero_grad()
                loss.backward()
                optimizer_sub.step()
            
            shrunk_coef_sub = model_en_sub.coef.detach().numpy()
            
            resid_sub = Y_trunc_sub - X_lagged_sub @ shrunk_coef_sub.T
            sigma_u_sub = np.cov(resid_sub.T)
            
            fevd_sub = generalized_fevd_custom(shrunk_coef_sub, sigma_u_sub, horizon, k, lag_order) * 100
            
            #Calculate and evaluate TSI
            diag_sub = np.diag(fevd_sub)
            TSI = (np.sum(fevd_sub) - np.sum(diag_sub)) / np.sum(fevd_sub) * 100
            
            sub_series.append(TSI)
        except Exception as e:
            print(f"Errore in robustness window {w}, start {start}: {e}")
            sub_series.append(np.nan)
    
    mean_tsi = np.nanmean(sub_series) if sub_series else np.nan
    var_tsi = np.nanvar(sub_series) if sub_series else np.nan
    print(f"Window={w}: Mean TSI={mean_tsi:.2f}%, Variazione (var)={var_tsi:.2f}")