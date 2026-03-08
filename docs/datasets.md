# Datasets for the General Unified World Model

Priority-ordered datasets for filling under-represented schema layers.

## Priority 1: World Bank WDI (`pip install wbgapi`)
- 17,500+ indicators across demographics, health, education, governance, environment, trade, technology
- 200+ economies, annual data
- `import wbgapi as wb; df = wb.data.DataFrame("SP.POP.TOTL")`

## Priority 2: OWID CO2/Energy (zero-friction CSV)
- CO2 emissions, energy mix, population, GDP for 200+ countries from 1750
- `pd.read_csv("https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv")`
- Energy: `pd.read_csv("https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv")`

## Priority 3: QoG Standard Dataset
- 2,100 pre-harmonized governance variables (V-Dem, WGI, Freedom House, Polity combined)
- Country-year panel 1946–2025, free CSV download
- https://www.gu.se/en/quality-government/qog-data

## Priority 4: UCDP + ACLED (conflict/military)
- UCDP: Armed conflicts 1946–present, georeferenced events, battle deaths
- ACLED: Real-time political violence/protest events, weekly updates (`pip install acled`)
- https://ucdp.uu.se/ and https://acleddata.com/

## Priority 5: IMF WEO (forecasts layer)
- GDP growth, inflation, unemployment forecasts for 190 countries
- `pip install weo; weo.download(year=2024, release="Oct")`
- Maps directly to forecasts.gdp_growth, forecasts.inflation, forecasts.policy_rate

## Priority 6: Open-Meteo (environmental, no auth)
- 80+ years hourly weather, air quality, no API key needed
- https://open-meteo.com/

## Priority 7: GDELT (information/media layer)
- World news events 1979–present, 15-min updates, 100+ languages
- `pip install gdelt`
- Complements existing news embeddings

## Priority 8: UN Population Prospects
- Population, fertility, mortality, migration for 237 countries, 1950–2100
- https://population.un.org/wpp/downloads

## Additional Sources

### Climate/Environmental
- **ERA5 Reanalysis**: `pip install cdsapi`, hourly global climate 1940–present
- **NASA GISTEMP**: Monthly temperature anomalies 1880–present (direct CSV)
- **OpenAQ**: Real-time air quality (`pip install py-openaq`)
- **EM-DAT**: Disaster database 1900–present (free registration)

### Political/Governance
- **V-Dem**: 400+ democracy indicators, 202 countries, 1789–2023
- **WGI**: 6 governance dimensions, 200+ economies, 1996–2024 (`wbgapi`)
- **Polity5**: Autocracy-democracy scores 1800–2018
- **Freedom House**: Political rights/civil liberties 1973–2024
- **Fragile States Index**: 12 conflict risk indicators 2006–2024

### Conflict/Military
- **SIPRI**: Military expenditure 1949–2024, arms transfers 1950–2025
- **Correlates of War**: Interstate wars, disputes, alliances 1816–2014

### Health/Demographic
- **WHO GHO**: 1,000+ health indicators, OData API
- **IHME GBD**: 292 causes of death, 375 diseases, 204 countries
- **UN Population Prospects**: Fertility, mortality, migration 1950–2100

### Trade/Economic
- **UN Comtrade**: Bilateral trade, 6,000+ commodities (`pip install comtradeapicall`)
- **OEC**: Economic Complexity Index (`pip install oec`)
- **Penn World Table**: Real GDP, productivity, 185 countries, 1950–2023
- **IMF WEO**: Forecasts via `pip install weo`
- **ILOSTAT**: Labor statistics, REST API

### Multi-Source
- **sdmx1**: Single API for World Bank, IMF, ECB, Eurostat, OECD, etc. (`pip install sdmx1`)
- **OWID Catalog**: Curated cross-referenced datasets (`pip install owid-catalog`)
