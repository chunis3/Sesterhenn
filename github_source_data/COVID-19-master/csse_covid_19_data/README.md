# JHU CSSE COVID-19 Dataset

## Table of contents

 * [Daily reports (csse_covid_19_daily_reports)](#daily-reports-csse_covid_19_daily_reports)
 * [USA daily state reports (csse_covid_19_daily_reports_us)](#usa-daily-state-reports-csse_covid_19_daily_reports_us)
 * [Time series summary (csse_covid_19_time_series)](#time-series-summary-csse_covid_19_time_series)
 * [Data modification records](#data-modification-records)
 * [UID Lookup Table Logic](#uid-lookup-table-logic)
---

## [Daily reports (csse_covid_19_daily_reports)](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports)

This folder contains daily case reports. All timestamps are in UTC (GMT+0).

### File naming convention
MM-DD-YYYY.csv in UTC.

### Field description
* <b>FIPS</b>: US only. Federal Information Processing Standards code that uniquely identifies counties within the USA.
* <b>Admin2</b>: County name. US only.
* <b>Province_State</b>: Province, state or dependency name.
* <b>Country_Region</b>: Country, region or sovereignty name. The names of locations included on the Website correspond with the official designations used by the U.S. Department of State.
* <b>Last Update</b>: MM/DD/YYYY HH:mm:ss  (24 hour format, in UTC).
* <b>Lat</b> and <b>Long_</b>: Dot locations on the dashboard. All points (except for Australia) shown on the map are based on geographic centroids, and are not representative of a specific address, building or any location at a spatial scale finer than a province/state. Australian dots are located at the centroid of the largest city in each state.
* <b>Confirmed</b>: Confirmed cases include presumptive positive cases  and probable cases, in accordance with CDC guidelines as of April 14.
* <b>Deaths</b>: Death totals in the US include confirmed and probable, in accordance with [CDC](https://www.cdc.gov/coronavirus/2019-ncov/cases-updates/cases-in-us.html) guidelines as of April 14.
* <b>Recovered</b>: Recovered cases outside China are estimates based on local media reports, and state and local reporting when available, and therefore may be substantially lower than the true number. US state-level recovered cases are from [COVID Tracking Project](https://covidtracking.com/).
* <b>Active:</b> Active cases = total confirmed - total recovered - total deaths.
* <b>Incidence_Rate</b>: Admin2 + Province_State + Country_Region.
* <b>Case-Fatality Ratio (%)</b>: = confirmed cases per 100,000 persons.
* <b>US Testing Rate</b>: = total test results per 100,000 persons. The "total test results" is equal to "Total test results
(Positive + Negative)" from [COVID Tracking Project](https://covidtracking.com/).
* <b>US Hospitalization Rate (%)</b>: = Total number hospitalized / Number confirmed cases. The "Total number hospitalized" is the "Hospitalized – Cumulative" count from [COVID Tracking Project](https://covidtracking.com/). The "hospitalization rate" and "hospitalized - Cumulative" data is only presented for those states which provide cumulative hospital data.

### Update frequency
* Files on and after April 23, once per day between 03:30 and 04:00 UTC.
* Files from February 2 to April 22: once per day around 23:59 UTC.
* Files on and before February 1: the last updated files before 23:59 UTC. Sources: [archived_data](https://github.com/CSSEGISandData/COVID-19/tree/master/archived_data) and dashboard.

### Data sources
Refer to the [mainpage](https://github.com/CSSEGISandData/COVID-19).

### Why create this new folder?
1. Unifying all timestamps to UTC, including the file name and the "Last Update" field.
2. Pushing only one file every day.
3. All historic data is archived in [archived_data](https://github.com/CSSEGISandData/COVID-19/tree/master/archived_data).

---
## [USA daily state reports (csse_covid_19_daily_reports_us)](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports_us)

This table contains an aggregation of each USA State level data.

### File naming convention
MM-DD-YYYY.csv in UTC.

### Field description
* <b>Province_State</b> - The name of the State within the USA.
* <b>Country_Region</b> - The name of the Country (US).
* <b>Last_Update</b> - The most recent date the file was pushed.
* <b>Lat</b> - Latitude.
* <b>Long_</b> - Longitude.
* <b>Confirmed</b> - Aggregated confirmed case count for the state.
* <b>Deaths</b> - Aggregated Death case count for the state.
* <b>Recovered</b> - Aggregated Recovered case count for the state.
* <b>Active</b> - Aggregated confirmed cases that have not been resolved (Active = Confirmed - Recovered - Deaths).
* <b>FIPS</b> - Federal Information Processing Standards code that uniquely identifies counties within the USA.
* <b>Incident_Rate</b> - confirmed cases per 100,000 persons.
* <b>People_Tested</b> - Total number of people who have been tested.
* <b>People_Hospitalized</b> - Total number of people hospitalized.
* <b>Mortality_Rate</b> - Number recorded deaths * 100/ Number confirmed cases.
* <b>UID</b> - Unique Identifier for each row entry. 
* <b>ISO3</b> - Officialy assigned country code identifiers.
* <b>Testing_Rate</b> - Total number of people tested per 100,000 persons.
* <b>Hospitalization_Rate</b> - Total number of people hospitalized * 100/ Number of confirmed cases.

### Update frequency
* Once per day between 03:30 and 04:00 UTC.

### Data sources
Refer to the [mainpage](https://github.com/CSSEGISandData/COVID-19).

---
## [Time series summary (csse_covid_19_time_series)](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series)

See [here](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/README.md).

---
## Data modification records
We are also monitoring the curve change. Any errors made by us will be corrected in the dataset. Any possible errors from the original data sources will be listed here as a reference.
* NHC 2/14: Hubei Province deducted 108 prior deaths from the death toll due to double counting.
* For Hubei Province: from Feb 13 (GMT +8), we report both clinically diagnosed and lab-confirmed cases. For lab-confirmed cases only (Before Feb 17), please refer to [who_covid_19_situation_reports](https://github.com/CSSEGISandData/COVID-19/tree/master/who_covid_19_situation_reports). 
* On Feb 27 Italy made a change in their testing protocols, to limit coronavirus testing to at-risk people showing symptoms of COVID-19. ([Source](https://apnews.com/6c7e40fbec09858a3b4dbd65fe0f14f5))
* About DP 3/1: All cases of COVID-19 in repatriated US citizens from the Diamond Princess are grouped together, and their location is currently designated at the ship’s port location off the coast of Japan. These individuals have been assigned to various quarantine locations (in military bases and hospitals) around the US. This grouping is consistent with the CDC.
* Hainan Province active cases update (4/13): We responded to the error from 3/24 to 4/1 we had incorrect data for Hainan Province.  We had -6 active cases (168 6 168 -6). We applied the correction (168 6 162 0) that was applied on 4/2 for this period (3/24 to 4/1).
* Florida in the daily report US (4/13): Source data error. Correction 123,019 -> 21,019.
* Okaloosa, Florida in the dail report (4/13): Source data error. Correction 102,103 -> 103.
* The death toll in Wuhan was revised from 2579 to 3869 (4/17). ([Source1](http://www.china.org.cn/china/Off_the_Wire/2020-04/17/content_75943843.htm), [Source2](http://www.nhc.gov.cn/yjb/s7860/202004/51706a79b1af4349b99264420f2cee54.shtml))
* About France confirmed cases (4/16): after communicating with solidarites-sante.gouv.fr, we decided to make these adjustments based on public available information. From April 4 to April 11, only "cas confirmés" are counted as confirmed cases in our dashboard. Starting from April 12, both "cas confirmés" and "cas possibles en ESMS" (probable cases from ESMS) are counted into confirmed cases in our dashboard. ([More details](https://github.com/CSSEGISandData/COVID-19/issues/2094))
* Benton and Franklin, WA on April 21 and 22. Data were adjusted/added to match the WA DOH report. See [errata](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/Errata.csv) for details.
* April 22, cases within the Navajo Nation had been tracked as an independent data source which resulted in double counting of the cases and deaths within Arizona, New Mexico, and Nevada. The US time series files for confirmed from 4/1 and 4/8 and the US time series files for deaths from 3/31 to 4/17 were corrected to remove the double counting. Adjustments were also made for Navajo County, AZ; Cococino County, AZ; Apache County, AZ; San Juan County, NM; McKinley County, NM; Cibola County, NM; Socorrco County, NM; and San Juan County, UT. See errata file for specfic details.
* April 24, time_series_covid19_deaths_us.csv for New York City, NY adjusted to add back-distribute dated probable deaths. time_series_covid19_confirmed_us.csv for New York City, NY adjusted to remove probable deaths as cases. This change is in line with CDC reporting guidelines.
* April 26, recovered data for Australian territories from 4/20 to 4/26 had gone stale. Historical values from new source used to fill in the six day gap.
* April 28, for consistency, we no longer report the hospitalization data as the max of "current - hospitalized" and "cumulative - hospitalized", and instead only report 'cumulative - hospitalized' from [Covid Tracking Project](https://covidtracking.com/). For states that do not provide cumulative hospital counts no hospital data will be shown.
* April 28, Lithuania: The number of confirmed infection cases. Until 28 April, information has been provided on positive laboratory test results rather than on positive cases (people). ([Source](https://lietuva.lt/wp-content/uploads/2020/04/UPDATE-April-28.pdf)).
* April 30, all death values for the United Kingdom adjusted due to the release of deaths in care homes.
* May 1, all data for Kosovo and Serbia from 4/19-4/30 adjusted due to stale data source.
* May 2, clarification of the handling of data in France ([GitHub Issue](https://github.com/CSSEGISandData/COVID-19/issues/2459)).
* May 15, clairification of the handling of data for Spain ([GitHub Issue](https://github.com/CSSEGISandData/COVID-19/issues/2522)).
* May 20, the drop of cumulative confirmed cases in the UK. "This is due to historical data revisions across all pillars." ([Source](https://www.gov.uk/guidance/coronavirus-covid-19-information-for-the-public), [DHSCgovuk Twitter](https://twitter.com/DHSCgovuk/status/1263159710892638208)).
* May 27, removal of recovered data from Netherlands due to lack of data reporting by national health ministry.
* June 2, France: Reduction in confirmed cases due to a change in calculation method. Since June 2, patients who test positive are only counted once. (Baisse des cas confirmés due à un changement de méthode de calcul. Depuis le 2 juin, les patients testés positifs ne sont plus comptés qu’une seule fois.) ([Source](https://dashboard.covid19.data.gouv.fr/vue-d-ensemble?location=FRA))
* June 5, On June 2nd, Chile’s Ministerio de Salud began reporting national “total active cases” where in the past they had reported national “total recoveries”.  To accommodate this change and to stay consistent with the ministry’s reporting of active cases, from June 2nd forward we are computing recoveries based on the formula “Active Cases = Total Case – Deaths – Recoveries”.  Based on this, the data for Chile will reflects a jump in recoveries on June 2nd. ([Source](https://www.minsal.cl/nuevo-coronavirus-2019-ncov/casos-confirmados-en-chile-covid-19/))
* June 5, In an internal audit of the data for Sweden, it has become clear to our team that our reported total of recoveries conflates regional reporting of the number of patients being released from hospitals with country wide recovery data.  As this regional reporting is not universally available and represents only a subset of recoveries, our prior reporting did not accurately represent nationwide recoveries.  To ensure the accuracy of our data, we have chosen to nullify the number of recovered cases in Sweden until the data is released by the national health ministry. We will also be removing recovery data from our historical time series due to this assessment.
* June 5, As noted in the disclaimer for the dashboard, the geographic designations in this data have been designed to be consistent with public guidance from the US State Department.  This does not imply the expression of any opinion whatsoever on the part of JHU concerning the legal status of any country, area or territory or of its authorities.  In implementing subnational data for the Russian Federation and the Ukraine, data for the Crimean Peninsula has been apportioned in line with this guidance.  This adjustment explains a difference in national totals for both the Russian Federation and Ukraine relative to alternate reporting.
* June 10, Our previous reporting for Pakistan had a single day delay. A recent update corrected this issue but resulted in data for June 7th being lost. We have corrected this issue by adding June 7th manually and pulling all of the Pakistan data back by a single day.
* June 11, Michigan, US. Michigan started to report probable cases and probable deaths on June 5. ([Source](https://www.michigan.gov/coronavirus/0,9753,7-406-98158-531156--,00.html)) We combined the probable cases into the confirmed cases, and the probable deaths into the deaths. As a consequence, a spike with 5.5k+ cases is shown in our daily cases bar chart.
* June 13, Through data provided by the Michigan Department of Health and Human Service’s (MDHHS) Communicable Disease Division, we were able to appropriately distribute the probable cases MDHHS began reporting on June 5th.
* June 12, Louis City, MO, data for confirmed cases and deaths from March 16 to June 11 were updated to match up with the updated official report at the [City of St. Louis dashboard](https://www.stlouis-mo.gov/covid-19/data/index.cfm). Date of the first case was updated to March 16, and date of the first deaths was updated to March 23.
* June 12, Louis County, MO, data for confirmed cases and deaths from March 9 to June 11 were updated to match up with the updated official report at [St. Louis County government site](https://stlcorona.com/resources/covid-19-statistics1/). Date of the first case was remained on March 8, and date of the first deaths was updated to March 20.
* June 12, Massachusetts, confirmed cases on April 15 and from April 17 to June 11 were updated to match up with the updated official report from the [Massachusetts government raw data - County.csv](https://www.mass.gov/info-details/covid-19-response-reporting). No spike any more on June 1, since all probable cases are added back. Dukes and Nantucket are still reported together, though County.csv lists them separately.
* June 16th, delay in reporting from Oregon Health Authority resulted in time series for confirmed and deaths not updating for June 14th. Updated via data from this [report](https://www.oregon.gov/oha/ERD/Pages/Oregon-reports-101-new-confirmed-presumptive-COVID-19-cases-2-new-deaths.aspx). Recovered data was not available for this date.
* June 19th, cases data for Belarus on April 18th and 19th were adjusted. Initial error was due to a [delay] (https://news.tut.by/society/681391.html) in reporting by the Belarusian health authorities that wasn't properly distributed.
* Not a data modification, but we wish to draw attention to [issue #2722](https://github.com/CSSEGISandData/COVID-19/issues/2722) that explains the recent spike in cases in Chile.
* June 25, NJ began reporting probable deaths today and the record for the 25th reflects these 1854 deaths not previously reported.  Additional information can be found in the [transcript](https://nj.gov/governor/news/news/562020/approved/20200625a.shtml) of the state's June 25th coronavirus briefing.
* June 27, internal audit identified issue with calculation of probable cases in nursing homes for France. The French Health Ministry ended public reporting of this number on June 1st - we have since carried that number of probable cases forward.

---
## [UID Lookup Table Logic](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv)

1.	All countries without dependencies (entries with only Admin0).
  *	None cruise ship Admin0: UID = code3. (e.g., Afghanistan, UID = code3 = 4)
  *	Cruise ships in Admin0: Diamond Princess UID = 9999, MS Zaandam UID = 8888.
2.	All countries with only state-level dependencies (entries with Admin0 and Admin1).
  *	Demark, France, Netherlands: mother countries and their dependencies have different code3, therefore UID = code 3. (e.g., Faroe Islands, Denmark, UID = code3 = 234; Denmark UID = 208)
  *	United Kingdom: the mother country and dependencies have different code3s, therefore UID = code 3. One exception: Channel Islands is using the same code3 as the mother country (826), and its artificial UID = 8261.
  *	Australia: alphabetically ordered all states, and their UIDs are from 3601 to 3608. Australia itself is 36.
  *	Canada: alphabetically ordered all provinces (including cruise ships and recovered entry), and their UIDs are from 12401 to 12415. Canada itself is 124.
  *	China: alphabetically ordered all provinces, and their UIDs are from 15601 to 15631. China itself is 156. Hong Kong, Macau and Taiwan have their own code3.
  *	Germany: alphabetically ordered all admin1 regions (including Unknown), and their UIDs are from 27601 to 27617. Germany itself is 276.
  * Italy: UIDs are combined country code (380) with `codice_regione`, which is from [Dati COVID-19 Italia](https://github.com/pcm-dpc/COVID-19). Exceptions: P.A. Bolzano is 38041 and P.A. Trento is 38042.
3.	The US (most entries with Admin0, Admin1 and Admin2).
  *	US by itself is 840 (UID = code3).
  *	US dependencies, American Samoa, Guam, Northern Mariana Islands, Virgin Islands and Puerto Rico, UID = code3. Their FIPS codes are different from code3.
  *	US states: UID = 840 (country code3) + 000XX (state FIPS code). Ranging from 8400001 to 84000056.
  *	Out of [State], US: UID = 840 (country code3) + 800XX (state FIPS code). Ranging from 8408001 to 84080056.
  *	Unassigned, US: UID = 840 (country code3) + 900XX (state FIPS code). Ranging from 8409001 to 84090056.
  *	US counties: UID = 840 (country code3) + XXXXX (5-digit FIPS code).
  *	Exception type 1, such as recovered and Kansas City, ranging from 8407001 to 8407999.
  *	Exception type 2, only the New York City, which is replacing New York County and its FIPS code.
  *	Exception type 3, Diamond Princess, US: 84088888; Grand Princess, US: 84099999.
4. Population data sources.
 * United Nations, Department of Economic and Social Affairs, Population Division (2019). World Population Prospects 2019, Online Edition. Rev. 1. https://population.un.org/wpp/Download/Standard/Population/
 * eurostat: https://ec.europa.eu/eurostat/web/products-datasets/product?code=tgs00096
 * The U.S. Census Bureau: https://www.census.gov/data/datasets/time-series/demo/popest/2010s-counties-total.html
 * Mexico population 2020 projection: [Proyecciones de población](http://sniiv.conavi.gob.mx/(X(1)S(kqitzysod5qf1g00jwueeklj))/demanda/poblacion_proyecciones.aspx?AspxAutoDetectCookieSupport=1)
* Brazil 2019 projection: ftp://ftp.ibge.gov.br/Estimativas_de_Populacao/Estimativas_2019/
* Peru 2020 projection: https://www.citypopulation.de/en/peru/cities/
* India 2019 population: http://statisticstimes.com/demographics/population-of-indian-states.php

Disclaimer: \*The names of locations included on the Website correspond with the official designations used by the U.S. Department of State. The presentation of material therein does not imply the expression of any opinion whatsoever on the part of JHU concerning the legal status of any country, area or territory or of its authorities. The depiction and use of boundaries, geographic names and related data shown on maps and included in lists, tables, documents, and databases on this website are not warranted to be error free nor do they necessarily imply official endorsement or acceptance by JHU.
