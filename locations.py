# Getting the village locations using Google Maps API
import googlemaps
from os import path
import pandas as pd

def location(villages_name, google_key, save_path="./data/geolocation_data", name="villages_location.csv"):
    """
    Gets the location of the villages
    """
    if not path.isfile(save_path+"/"+name):
        gmaps = googlemaps.Client(key=google_key)
        geocode_result = {village : gmaps.geocode(village + " Senegal")[0]['geometry']['location'] for village in villages_name \
                          if len(gmaps.geocode(village + " Senegal"))>0} # may take a while (~ up to 20 minutes
        
        locations=pd.DataFrame(geocode_result).T
        locations.reset_index(level=0, inplace=True)
        locations.rename(columns={"index": "villages"}, inplace=True)
        locations.to_csv(save_path+"/"+name, index=False)
        return locations
    return pd.read_csv(save_path+"/"+name)

    
    