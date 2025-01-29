# DistanceToRTTMapping
exploring the relationship between Round Trip Time(RTT) and and Distance in KM, using the free RIPE ATLAS API and python. Retracing the steps of the following older paper:
> [**Spotter: A model based active geolocation service**]([https://doi.org/10.1109/INFCOM.2011.5935165](https://doi.org/10.1109/INFCOM.2011.5935165))  
> Laki, Sándor and Mátray, Péter and Hága, Péter and Sebők, Tamás and Csabai, István and Vattay, Gábor <br>
> *2011 Proceedings IEEE INFOCOM*

The purpose of this repsitory is to validate/compare the observations made about the ping vs distance relationship in the Spotter paper. here is a comparison of the results:
### Recreated
![screenshot](results.png)
## Requirements
- Anaconda
- 500mb free space for anaconda env and data from RIPE ATLAS API

## Usage
open terminal with anaconda activated. Then execute:
```
conda create -n distance_rtt python=3.11
conda activate distance_rtt
pip install -r requirements.txt
```
Then, to execute the script, simply run:
```
python main.py
```
