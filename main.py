import geopy.distance
import requests
import requests_cache
import numpy as np
import os
from enum import Enum
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, scale

from sklearn.pipeline import make_pipeline
from scipy import stats

class ASN_relationship(Enum):
    SAME = 0
    NEIGHBOURS = 1
    STRANGERS = 2

results = []

ping_outputs = {}

#chosen at random from https://atlas.ripe.net/measurements/public
measurement_ids = [86322689,80079508,55057518,74279978,55065549,47701605,29317880,29318407]

#the api calls are the same every time and there is a lot of data so we make an effort to cache results.
if os.path.exists("cached_results.npy"):
    results = np.load("cached_results.npy")
else:
    with requests_cache.enabled():
        for id in measurement_ids:
            ping_outputs[id] = {}
            ping_outputs[id]["results"] = requests.get(f"https://atlas.ripe.net/api/v2/measurements/{id}/latest/").json()
            probe_id =[each for each in requests.get(f"https://atlas.ripe.net/api/v2/measurements/{id}/").json()["tags"] if each.isnumeric()][0]
            ping_outputs[id]["prb_id"] = int(probe_id)

    
    for measurement_id in ping_outputs:
        T = None
        with requests_cache.enabled():
            probe_id = ping_outputs[measurement_id]["prb_id"]
            T= requests.get(f"https://atlas.ripe.net/api/v2/probes/{probe_id}/").json()
        
        T_ASN = T["asn_v4"]
        
        for pair in ping_outputs[measurement_id]["results"]:
            L= None
            with requests_cache.enabled():
                probe_id = pair["prb_id"]
                L= requests.get(f"https://atlas.ripe.net/api/v2/probes/{probe_id}/").json()


            #we have to swap coordinate positions here because geojson stores long lat and geopy thinks otherwise
            distance = geopy.distance.distance((T["geometry"]["coordinates"][1], T["geometry"]["coordinates"][0]),
                                                (L["geometry"]["coordinates"][1], L["geometry"]["coordinates"][0]))
            #get lowest round trip time (RTT)
            if "min" in pair:
                min_rtt = pair["min"]
                #some entries are broken for whatever reason
                if min_rtt < 0.2:
                    print(min_rtt)
                    continue
            else:
                continue
            #get ASN classification
            L_ASN = L["asn_v4"]

            AS_relation = None
            if L_ASN == T_ASN:
                AS_relation = ASN_relationship.SAME.value
            else:
                T_neighbours = []
                with requests_cache.enabled():
                    response = requests.get("https://stat.ripe.net/data/asn-neighbours/data.json",
                                                    params={
                                                        "resource": f"AS{T_ASN}",
                                                        "query_time": pair["timestamp"]
                                                    }).json()
                    T_neighbours = [result["asn"] for result in response["data"]["neighbours"]]
                if L_ASN in T_neighbours:
                    AS_relation = ASN_relationship.NEIGHBOURS.value
                else:
                    AS_relation = ASN_relationship.STRANGERS.value

            results.append([distance.km, min_rtt, AS_relation])
            if len(results) > 20000:
                break

            if len(results) % 500 == 0:
                print(len(results))
    np.save("cached_results.npy",results)

#Following method of Spotter paper, we will filter out and values
parsed = np.array(results)
parsed = parsed[parsed[:, 1] < 55]
y = parsed[:, 0]
X = parsed[:, 1].reshape(-1, 1)

# put it all into nice dataframe
# data = pd.DataFrame({
#     'x': parsed[:, 1],
#     'AS_relation': parsed[:, 2],
#     'y': parsed[:, 0]
# })

# Explicitly define categories
# data['AS_relation'] = pd.Categorical(data['AS_relation'], categories=[0.0, 1.0, 2.0])

# Fit polynomial using all data
poly = PolynomialFeatures(degree=6, include_bias=True)

# Robustly fit linear model with RANSAC algorithm
# poly_ransac_model = make_pipeline(poly, linear_model.RANSACRegressor(random_state=42))
# poly_ransac_model.fit(X, y)

model = linear_model.LinearRegression()
model.fit(X,y) 
# Predict data of estimated models
line_X = np.linspace(X.min(), 50, 1000)[:, np.newaxis]
line_y = np.linspace(y.min(), y.max(), len(line_X))[:, np.newaxis]
line_y_hat = model.predict(line_X)

y_pred = model.predict(X)

def convert_to_probability_distribution(y_actual):
    # Calculate mean and standard deviation
    mean = np.mean(y_actual)
    std = np.std(y_actual)
    
    # Convert to z-scores
    z_scores = (y_actual - mean) / std
    
    # Calculate probability density
    probabilities = stats.norm.pdf(z_scores)
    
    return z_scores, probabilities


# Create a figure with 3 subplots in a row
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# First plot: Scatter plot with regression line
axes[0].scatter(X, y, marker=".")
axes[0].plot(line_X, line_y_hat, color="red", linewidth=2, label="best fit line")
axes[0].legend(loc="lower right")
axes[0].set_xlabel("Round Trip Time (RTT)")
axes[0].set_ylabel("Distance")

# Second plot: Probability distribution
z_scores, probabilities = convert_to_probability_distribution(y)
axes[1].plot(z_scores, probabilities, 'b.')
axes[1].set_xlabel('Z-scores')
axes[1].set_ylabel('Probability Density')
axes[1].set_title('Standardized Probability Distribution')
axes[1].grid(True)

# Third plot: Normal probability plot
axes[2].scatter(scale(y), scale(model.predict(X)))
axes[2].plot([-3, 3], [-3, 3], 'r--')  # Reference line
axes[2].set_xlabel('Standardized Distance Values')
axes[2].set_ylabel('Standardized Expected Distance Values')
axes[2].set_title('Normal Probability Plot')
axes[2].grid(True)

# Adjust layout and display
plt.tight_layout()
plt.show()

