import os
import csv
import requests
import urllib
import json
import losser.losser as losser

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets/")
pli_url = "https://data.wprdc.org/datastore/dump/4e5374be-1a88-47f7-afee-6a79317019b4"
property_url = "https://data.wprdc.org/dataset/2b3df818-601e-4f06-b150-643557229491/resource/2514a4e4-5842-4dca-aff6-099bcd68482c/download/assessments.csv"
parcel_url = "https://data.wprdc.org/dataset/2536e5e2-253b-4c58-969d-687828bb94c6/resource/4b68a6dd-b7ea-4385-b88e-e7d77ff0b294/download/parcelcentroidaug102016.csv"
tax_url = "https://data.wprdc.org/api/action/datastore_search_sql?sql=SELECT * from \"65d0d259-3e58-49d3-bebb-80dc75f61245\" WHERE tax_year > 2008"

print(dir_path)

print("Getting pli...")
pli_response = requests.get(pli_url)
with open(os.path.join(dir_path, "pli.csv"), 'wb') as f:
    f.write(pli_response.content)

print("Getting pittdata...")
property_response = requests.get(property_url)
with open(os.path.join(dir_path, "pittdata.csv"), 'wb') as f:
    f.write(property_response.content)

print("Getting parcels...")
parcel_response = requests.get(parcel_url)
with open(os.path.join(dir_path, "parcels.csv"), 'wb') as f:
    f.write(parcel_response.content)

print("Getting tax liens...")
tax_response = urllib.urlopen(tax_url)
tax_dict = json.loads(tax_response.read())['result']['records']
columns = "columns.json"
taxdata = losser.table(tax_dict, columns)
with open(os.path.join(dir_path, "tax.csv"), 'wb') as outfile:
    fp = csv.DictWriter(outfile, taxdata[0].keys())
    fp.writeheader()
    fp.writerows(taxdata)
