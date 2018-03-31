# fire_risk_residential

This project extends the previous [fire_risk_analysis](https://github.com/CityofPittsburgh/fire_risk_analysis) from predicting fire risk in commercial properties to predicting fire risk in residential properties, in order to inform the community risk reduction efforts. This is the legacy version. The most up to date version is under the [development branch](https://github.com/Eve-Chen/fire_risk_residential/tree/development)

## How to set up
1. Run `getdata.py` to scrape [WPRDC](https://www.wprdc.org) for:
- City of Pittsburgh property data ("pittdata.csv")
- City of Pittsburgh parcel data ("parcels.csv")
- Permits, Licenses, and Inspections data ("pli.csv")
- Tax lien data ('tax.csv')

2. Download "acs_income.csv", "acs_occupancy.csv", "acs_year_built.csv", "acs_year_moved.csv" from google drive and put them in the `datasets` folder.

3. Run `risk_model_residential_kdd.py`.
