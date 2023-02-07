-- Extract heartrate, respiratory rate, O2 Saturation and temperature from vital table
-- based on: https://github.com/MIT-LCP/eicu-code/blob/master/concepts/pivoted/pivoted-vital.sql

DROP TABLE IF EXISTS pivoted_heartrate CASCADE;
CREATE TABLE pivoted_heartrate as
-- create columns with only numeric data
with nc_heartrate as
(
select
    patientunitstayid
  , nursingchartoffset
  , nursingchartentryoffset
  , case
      when nursingchartcelltypevallabel = 'Heart Rate'
       and nursingchartcelltypevalname = 'Heart Rate'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as heartrate
  from eicu_crd.nursecharting
  -- speed up by only looking at a subset of charted data
  where nursingchartcelltypecat in
  (
    'Vital Signs','Scores','Other Vital Signs and Infusions'
  )
)
, vw_heartrate as (
	
		select
		  patientunitstayid
		, nursingchartoffset as chartoffset
		, avg(case when heartrate >= 25 and heartrate <= 225 then heartrate else null end) as heartrate
		from nc_heartrate
		WHERE heartrate IS NOT NULL and abs(nursingchartoffset) < 120
		group by patientunitstayid, nursingchartoffset
		order by patientunitstayid, nursingchartoffset

	)

	-- keep only the observation closest to ICU admission
	select vw_heartrate.patientunitstayid, minvalue as heartrate_chartoffset, heartrate
	from(
		select distinct patientunitstayid
		,min(ABS(chartoffset)) * case when
			(min(case when chartoffset > 0 then chartoffset end) > abs(max(case when chartoffset < 0 then chartoffset end))
			 or 
			 (min(case when chartoffset > 0 then chartoffset end) is NULL))
		 THEN -1 ELSE 1 END as minvalue -- extract minimum absolut chartoffset
		from vw_heartrate
		group by patientunitstayid) t1
	inner join vw_heartrate
	on vw_heartrate.patientunitstayid = t1.patientunitstayid
	and vw_heartrate.chartoffset = t1.minvalue;



DROP TABLE IF EXISTS pivoted_resprate CASCADE;
CREATE TABLE pivoted_resprate as
-- create columns with only numeric data
with nc_resprate as
(
select
    patientunitstayid
  , nursingchartoffset
  , nursingchartentryoffset
  , case
      when nursingchartcelltypevallabel = 'Respiratory Rate'
       and nursingchartcelltypevalname = 'Respiratory Rate'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as RespiratoryRate
  from eicu_crd.nursecharting
  -- speed up by only looking at a subset of charted data
  where nursingchartcelltypecat in
  (
    'Vital Signs','Scores','Other Vital Signs and Infusions'
  )
)
, vw_resprate as (
	
		select
		  patientunitstayid
		, nursingchartoffset as chartoffset
	    , avg(case when RespiratoryRate >= 0 and RespiratoryRate <= 60 then RespiratoryRate else null end) as RespiratoryRate
		from nc_resprate
		WHERE RespiratoryRate IS NOT NULL and abs(nursingchartoffset) < 120
		group by patientunitstayid, nursingchartoffset
		order by patientunitstayid, nursingchartoffset

	)

	-- keep only the observation closest to ICU admission
	select vw_resprate.patientunitstayid, minvalue as resprate_chartoffset, RespiratoryRate
	from(
		select distinct patientunitstayid
		,min(ABS(chartoffset)) * case when
			(min(case when chartoffset > 0 then chartoffset end) > abs(max(case when chartoffset < 0 then chartoffset end))
			 or 
			 (min(case when chartoffset > 0 then chartoffset end) is NULL))
		 THEN -1 ELSE 1 END as minvalue -- extract minimum absolut chartoffset
		from vw_resprate
		group by patientunitstayid) t1
	inner join vw_resprate
	on vw_resprate.patientunitstayid = t1.patientunitstayid
	and vw_resprate.chartoffset = t1.minvalue;




DROP TABLE IF EXISTS pivoted_o2sat CASCADE;
CREATE TABLE pivoted_o2sat as
-- create columns with only numeric data
with nc_o2sat as
(
select
    patientunitstayid
  , nursingchartoffset
  , nursingchartentryoffset
  , case
      when nursingchartcelltypevallabel = 'O2 Saturation'
       and nursingchartcelltypevalname = 'O2 Saturation'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as o2saturation
  from eicu_crd.nursecharting
  -- speed up by only looking at a subset of charted data
  where nursingchartcelltypecat in
  (
    'Vital Signs','Scores','Other Vital Signs and Infusions'
  )
)
, vw_o2sat as (
	
		select
		  patientunitstayid
		, nursingchartoffset as chartoffset
	    , avg(case when o2saturation >= 0 and o2saturation <= 100 then o2saturation else null end) as spo2
		from nc_o2sat
		WHERE o2saturation IS NOT NULL and abs(nursingchartoffset) < 120
		group by patientunitstayid, nursingchartoffset
		order by patientunitstayid, nursingchartoffset

	)

	-- keep only the observation closest to ICU admission
	select vw_o2sat.patientunitstayid, minvalue as o2sat_chartoffset, spo2
	from(
		select distinct patientunitstayid
		,min(ABS(chartoffset)) * case when
			(min(case when chartoffset > 0 then chartoffset end) > abs(max(case when chartoffset < 0 then chartoffset end))
			 or 
			 (min(case when chartoffset > 0 then chartoffset end) is NULL))
		 THEN -1 ELSE 1 END as minvalue -- extract minimum absolut chartoffset
		from vw_o2sat
		group by patientunitstayid) t1
	inner join vw_o2sat
	on vw_o2sat.patientunitstayid = t1.patientunitstayid
	and vw_o2sat.chartoffset = t1.minvalue;



DROP TABLE IF EXISTS pivoted_temp CASCADE;
CREATE TABLE pivoted_temp as
-- create columns with only numeric data
with nc_temp as
(
select
    patientunitstayid
  , nursingchartoffset
  , nursingchartentryoffset
  ,  case
      when nursingchartcelltypevallabel = 'Temperature'
       and nursingchartcelltypevalname = 'Temperature (C)'
       and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
       and nursingchartvalue not in ('-','.')
          then cast(nursingchartvalue as numeric)
      else null end
    as temperature
  from eicu_crd.nursecharting
  -- speed up by only looking at a subset of charted data
  where nursingchartcelltypecat in
  (
    'Vital Signs','Scores','Other Vital Signs and Infusions'
  )
)
, vw_temp as (
	
		select
		  patientunitstayid
		, nursingchartoffset as chartoffset
	    , avg(case when temperature >= 25 and temperature <= 46 then temperature else null end) as temperature
		from nc_temp
		WHERE temperature IS NOT NULL and abs(nursingchartoffset) < 120
		group by patientunitstayid, nursingchartoffset
		order by patientunitstayid, nursingchartoffset

	)

	-- keep only the observation closest to ICU admission
	select vw_temp.patientunitstayid, minvalue as temp_chartoffset, temperature
	from(
		select distinct patientunitstayid
		,min(ABS(chartoffset)) * case when
			(min(case when chartoffset > 0 then chartoffset end) > abs(max(case when chartoffset < 0 then chartoffset end))
			 or 
			 (min(case when chartoffset > 0 then chartoffset end) is NULL))
		 THEN -1 ELSE 1 END as minvalue -- extract minimum absolut chartoffset
		from vw_temp
		group by patientunitstayid) t1
	inner join vw_temp
	on vw_temp.patientunitstayid = t1.patientunitstayid
	and vw_temp.chartoffset = t1.minvalue;


-- Merge all the variable tables together

DROP TABLE IF EXISTS pivoted_vital CASCADE;
CREATE TABLE pivoted_vital as
select h.patientunitstayid, heartrate_chartoffset, heartrate
		, resprate_chartoffset, respiratoryrate
		, o2sat_chartoffset, spo2
		, temp_chartoffset, temperature
from pivoted_heartrate h
inner join pivoted_resprate r
on h.patientunitstayid = r.patientunitstayid
inner join pivoted_o2sat o
on h.patientunitstayid = o.patientunitstayid
inner join pivoted_temp t
on h.patientunitstayid = t.patientunitstayid





