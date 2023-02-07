-- extract Glasgow coma score
-- based on: https://github.com/MIT-LCP/eicu-code/blob/master/concepts/pivoted/pivoted-gcs.sql
drop table if exists pivoted_gcs cascade;
create table pivoted_gcs as
with nc as
	(
	select
	  patientunitstayid
	  , nursingchartoffset as chartoffset
	  , min(case
		  when nursingchartcelltypevallabel = 'Glasgow coma score'
		   and nursingchartcelltypevalname = 'GCS Total'
		   and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
		   and nursingchartvalue not in ('-','.')
			  then cast(nursingchartvalue as numeric)
		  when nursingchartcelltypevallabel = 'Score (Glasgow Coma Scale)'
		   and nursingchartcelltypevalname = 'Value'
		   and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
		   and nursingchartvalue not in ('-','.')
			  then cast(nursingchartvalue as numeric)
		  else null end)
		as gcs
	  , min(case
		  when nursingchartcelltypevallabel = 'Glasgow coma score'
		   and nursingchartcelltypevalname = 'Motor'
		   and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
		   and nursingchartvalue not in ('-','.')
			  then cast(nursingchartvalue as numeric)
		  else null end)
		as gcsmotor
	  , min(case
		  when nursingchartcelltypevallabel = 'Glasgow coma score'
		   and nursingchartcelltypevalname = 'Verbal'
		   and nursingchartvalue  ~ '^[-]?[0-9]+[.]?[0-9]*$'
		   and nursingchartvalue not in ('-','.')
			  then cast(nursingchartvalue as numeric)
		  else null end)
		as gcsverbal
	  , min(case
		  when nursingchartcelltypevallabel = 'Glasgow coma score'
		   and nursingchartcelltypevalname = 'Eyes'
		   and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
		   and nursingchartvalue not in ('-','.')
			  then cast(nursingchartvalue as numeric)
		  else null end)
		as gcseyes
	  from eicu_crd.nursecharting
	  -- speed up by only looking at a subset of charted data
	  where nursingchartcelltypecat in
	  (
		'Scores', 'Other Vital Signs and Infusions'
	  )
	  group by patientunitstayid, nursingchartoffset
	)
	-- apply some preprocessing to fields
	, ncproc AS
	(
	  select
		patientunitstayid
	  , chartoffset
	  , case when gcs > 2 and gcs < 16 then gcs else null end as gcs
	  , gcsmotor, gcsverbal, gcseyes
	  from nc
	)
	, vw0 as (
	
		select
		  patientunitstayid
		  , chartoffset
		  , gcs
		  , gcsmotor, gcsverbal, gcseyes
		FROM ncproc
		WHERE gcs IS NOT NULL
		OR gcsmotor IS NOT NULL
		OR gcsverbal IS NOT NULL
		OR gcseyes IS NOT NULL
		ORDER BY patientunitstayid
		
	)
	
	select vw0.patientunitstayid, minvalue as gcs_chartoffset , gcs
		  , gcsmotor, gcsverbal, gcseyes
	from(
		select distinct patientunitstayid
		,min(ABS(chartoffset)) * case when
			(min(case when chartoffset > 0 then chartoffset end) > abs(max(case when chartoffset < 0 then chartoffset end))
			 or 
			 (min(case when chartoffset > 0 then chartoffset end) is NULL))
		 THEN -1 ELSE 1 END as minvalue -- extract minimum absolut chartoffset
		from vw0
		group by patientunitstayid) t1
	inner join vw0
	on vw0.patientunitstayid = t1.patientunitstayid
	and vw0.chartoffset = t1.minvalue
	where abs(chartoffset) < 120
	