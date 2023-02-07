-- extract glucose and HCO3
-- based on: https://github.com/MIT-LCP/eicu-code/blob/master/concepts/pivoted/pivoted-lab.sql
-- remove duplicate labs if they exist at the same time

-- glucose
drop table if exists vw0_glucose cascade;
create table vw0_glucose as
	(
	  select
		  patientunitstayid
		, labname
		, labresultoffset
		, labresultrevisedoffset
	  from eicu_crd.lab
	  where labname in
	  (
		  
		'bedside glucose', 'glucose'
		
	  )
	  group by patientunitstayid, labname, labresultoffset, labresultrevisedoffset
	  having count(distinct labresult)<=1
	);
	
-- get the last lab to be revised
drop table if exists vw1_glucose cascade;
create table vw1_glucose as
	(
	  select
		  lab.patientunitstayid
		, lab.labname
		, lab.labresultoffset
		, lab.labresultrevisedoffset
		, lab.labresult
		, ROW_NUMBER() OVER
			(
			  PARTITION BY lab.patientunitstayid, lab.labname, lab.labresultoffset
			  ORDER BY lab.labresultrevisedoffset DESC
			) as rn
	  from eicu_crd.lab
	  inner join vw0_glucose
		ON  lab.patientunitstayid = vw0_glucose.patientunitstayid
		AND lab.labname = vw0_glucose.labname
		AND lab.labresultoffset = vw0_glucose.labresultoffset
		AND lab.labresultrevisedoffset = vw0_glucose.labresultrevisedoffset
	  -- only valid lab values
	  WHERE
			(lab.labname in ('bedside glucose', 'glucose') and lab.labresult >= 25 and lab.labresult <= 1500)
		
);
	
	
drop table if exists vw2_glucose cascade;
create table vw2_glucose as (
	
	select
		patientunitstayid
	  , labresultoffset as chartoffset
 	  , MAX(case when labname in ('bedside glucose', 'glucose') then labresult else null end) as glucose
	from vw1_glucose
	where rn = 1 and abs(labresultoffset) < 120
	group by patientunitstayid, labresultoffset
	order by patientunitstayid, labresultoffset
	
);
	

-- keep only the observation closest to ICU admission
drop table if exists pivoted_glucose cascade;
create table pivoted_glucose as (
	
	select vw2_glucose.patientunitstayid, minvalue as glucose_chartoffset, glucose
	from(
		select distinct patientunitstayid
		,min(ABS(chartoffset)) * case when
			(min(case when chartoffset > 0 then chartoffset end) > abs(max(case when chartoffset < 0 then chartoffset end))
			 or 
			 (min(case when chartoffset > 0 then chartoffset end) is NULL))
		 THEN -1 ELSE 1 END as minvalue -- extract minimum absolut chartoffset
		from vw2_glucose
		group by patientunitstayid) t1
	inner join vw2_glucose
	on vw2_glucose.patientunitstayid = t1.patientunitstayid
	and vw2_glucose.chartoffset = t1.minvalue
);


-- **************************************
-- bicarbonate **************************
drop table if exists vw0_bicarb cascade;
create table vw0_bicarb as
	(
	  select
		  patientunitstayid
		, labname
		, labresultoffset
		, labresultrevisedoffset
	  from eicu_crd.lab
	  where labname in
	  (
		  
		'bicarbonate' -- HCO3
		
	  )
	  group by patientunitstayid, labname, labresultoffset, labresultrevisedoffset
	  having count(distinct labresult)<=1
	);
	
-- get the last lab to be revised
drop table if exists vw1_bicarb cascade;
create table vw1_bicarb as
	(
	  select
		  lab.patientunitstayid
		, lab.labname
		, lab.labresultoffset
		, lab.labresultrevisedoffset
		, lab.labresult
		, ROW_NUMBER() OVER
			(
			  PARTITION BY lab.patientunitstayid, lab.labname, lab.labresultoffset
			  ORDER BY lab.labresultrevisedoffset DESC
			) as rn
	  from eicu_crd.lab
	  inner join vw0_bicarb
		ON  lab.patientunitstayid = vw0_bicarb.patientunitstayid
		AND lab.labname = vw0_bicarb.labname
		AND lab.labresultoffset = vw0_bicarb.labresultoffset
		AND lab.labresultrevisedoffset = vw0_bicarb.labresultrevisedoffset
	  -- only valid lab values
	  WHERE
			(lab.labname = 'bicarbonate' and lab.labresult >= 0 and lab.labresult <= 9999)
		
);
	
	
drop table if exists vw2_bicarb cascade;
create table vw2_bicarb as (
	
	select
		patientunitstayid
	  , labresultoffset as chartoffset
 	  , MAX(case when labname = 'bicarbonate' then labresult else null end) as bicarbonate
	from vw1_bicarb
	where rn = 1 and abs(labresultoffset) < 120
	group by patientunitstayid, labresultoffset
	order by patientunitstayid, labresultoffset
	
);
	

-- keep only the observation closest to ICU admission
drop table if exists pivoted_bicarb cascade;
create table pivoted_bicarb as (
	
	select vw2_bicarb.patientunitstayid, minvalue as bicarb_chartoffset, bicarbonate
	from(
		select distinct patientunitstayid
		,min(ABS(chartoffset)) * case when
			(min(case when chartoffset > 0 then chartoffset end) > abs(max(case when chartoffset < 0 then chartoffset end))
			 or 
			 (min(case when chartoffset > 0 then chartoffset end) is NULL))
		 THEN -1 ELSE 1 END as minvalue -- extract minimum absolut chartoffset
		from vw2_bicarb
		group by patientunitstayid) t1
	inner join vw2_bicarb
	on vw2_bicarb.patientunitstayid = t1.patientunitstayid
	and vw2_bicarb.chartoffset = t1.minvalue
);


select *
from pivoted_bicarb

-- join bicarbonate and glucose together
drop table if exists pivoted_lab0 cascade;
create table pivoted_lab0 as (
select g.patientunitstayid, glucose_chartoffset, glucose
		, bicarb_chartoffset, bicarbonate, b.patientunitstayid as patientunitstayid2
from pivoted_glucose g
full join pivoted_bicarb b
on g.patientunitstayid = b.patientunitstayid)


drop table if exists pivoted_lab cascade;
create table pivoted_lab as
select 
	case 
		when patientunitstayid is null then patientunitstayid2 else patientunitstayid end as patientunitstayid
		, glucose_chartoffset, glucose
		, bicarb_chartoffset, bicarbonate
from pivoted_lab0




