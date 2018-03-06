# fire_risk_residential

This project extends the previous [fire_risk_analysis](https://github.com/CityofPittsburgh/fire_risk_analysis) from predicting fire risk in commercial properties to predicting fire risk in residential properties, in order to inform the community risk reduction efforts.

## How to set up
1. Run `getdata.py` to scrape [WPRDC](https://www.wprdc.org) for:
- City of Pittsburgh property data ("pittdata.csv")
- City of Pittsburgh parcel data ("parcels.csv")
- Permits, Licenses, and Inspections data ("pli.csv")
- Tax lien data ('tax.csv')

2. Additional two datasets are used for predicting fire risk scores:
- Fire Incident data from PBF (public, aggregated version available at WPRDC. However, please note that due to privacy concerns, the most     detailed fire incident data that the model is trained on are not publicly accessible, but the aggregated version of the incident data       is available, at the block-level, instead of the address-level. At the moment, this script is not able to run on the aggregated, block-     level data.
- ACS data.

3. Run `risk_model_residential.py`.
