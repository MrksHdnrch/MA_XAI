-- based on: https://github.com/MIT-LCP/eicu-code/blob/master/concepts/pivoted/pivoted-bg.sql
-- get blood gas measures

-- pao2 ********************

drop table if exists pivoted_pao2 cascade;
create table pivoted_pao2 as

	
	with vw0 as
	(	
		  select
			  patientunitstayid
			, labname
			, labresultoffset
			, labresultrevisedoffset
		  from eicu_crd.lab
		  where labname in
		  (
				'paO2'
		  )
		  group by patientunitstayid, labname, labresultoffset, labresultrevisedoffset
		  having count(distinct labresult)<=1
	)
	-- get the last lab to be revised
	, vw1 as
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
		  inner join vw0
			ON  lab.patientunitstayid = vw0.patientunitstayid
			AND lab.labname = vw0.labname
			AND lab.labresultoffset = vw0.labresultoffset
			AND lab.labresultrevisedoffset = vw0.labresultrevisedoffset
		  WHERE
			 (lab.labname = 'paO2' and lab.labresult >= 15 and lab.labresult <= 720)
	), vw2 as (
		
			select
				patientunitstayid
			  , labresultoffset as chartoffset
			  -- the aggregate (max()) only ever applies to 1 value due to the where clause
			  , MAX(case when labname = 'paO2' then labresult else null end) as pao2
			from vw1
			where rn = 1 and abs(labresultoffset) < 120
			group by patientunitstayid, labresultoffset
			order by patientunitstayid, labresultoffset
	)
	
	-- keep only the observation closest to ICU admission
	select vw2.patientunitstayid, minvalue as pao2_chartoffset, pao2
	from(
		select distinct patientunitstayid
		,min(ABS(chartoffset)) * case when
			(min(case when chartoffset > 0 then chartoffset end) > abs(max(case when chartoffset < 0 then chartoffset end))
			 or 
			 (min(case when chartoffset > 0 then chartoffset end) is NULL))
		 THEN -1 ELSE 1 END as minvalue -- extract minimum absolut chartoffset
		from vw2
		group by patientunitstayid) t1
	inner join vw2
	on vw2.patientunitstayid = t1.patientunitstayid
	and vw2.chartoffset = t1.minvalue;
	

	
	-- paco2 ********************
	
	drop table if exists pivoted_paco2 cascade;
	create table pivoted_paco2 as

	-- https://github.com/MIT-LCP/eicu-code/blob/master/concepts/pivoted/pivoted-bg.sql
	-- get blood gas measures
	with vw0 as
	(	
		  select
			  patientunitstayid
			, labname
			, labresultoffset
			, labresultrevisedoffset
		  from eicu_crd.lab
		  where labname in
		  (
			 'paCO2'
		  )
		  group by patientunitstayid, labname, labresultoffset, labresultrevisedoffset
		  having count(distinct labresult)<=1
	)
	-- get the last lab to be revised
	, vw1 as
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
		  inner join vw0
			ON  lab.patientunitstayid = vw0.patientunitstayid
			AND lab.labname = vw0.labname
			AND lab.labresultoffset = vw0.labresultoffset
			AND lab.labresultrevisedoffset = vw0.labresultrevisedoffset
		  WHERE
		   (lab.labname = 'paCO2' and lab.labresult >= 5 and lab.labresult <= 250)

	), vw2 as (
		
			select
				patientunitstayid
			  , labresultoffset as chartoffset
			  -- the aggregate (max()) only ever applies to 1 value due to the where clause
			  , MAX(case when labname = 'paCO2' then labresult else null end) as paco2
			from vw1
			where rn = 1 and abs(labresultoffset) < 120
			group by patientunitstayid, labresultoffset
			order by patientunitstayid, labresultoffset
	)
	
	-- keep only the observation closest to ICU admission
	select vw2.patientunitstayid, minvalue as paco2_chartoffset, paco2
	from(
		select distinct patientunitstayid
		,min(ABS(chartoffset)) * case when
			(min(case when chartoffset > 0 then chartoffset end) > abs(max(case when chartoffset < 0 then chartoffset end))
			 or 
			 (min(case when chartoffset > 0 then chartoffset end) is NULL))
		 THEN -1 ELSE 1 END as minvalue -- extract minimum absolut chartoffset
		from vw2
		group by patientunitstayid) t1
	inner join vw2
	on vw2.patientunitstayid = t1.patientunitstayid
	and vw2.chartoffset = t1.minvalue;	
	
	
	-- Join pao2 and paco2 table
	drop table if exists pivoted_bg0 cascade;
	create table pivoted_bg0 as
	select p1.patientunitstayid, pao2_chartoffset, pao2, paco2_chartoffset, paco2, p2.patientunitstayid as patientunitstayid2
	from pivoted_pao2 p1
	full join pivoted_paco2 p2
	on p1.patientunitstayid = p2.patientunitstayid
	
	drop table if exists pivoted_bg cascade;
	create table pivoted_bg as
	select 
		case 
			when patientunitstayid is null then patientunitstayid2 else patientunitstayid end as patientunitstayid
			, pao2_chartoffset, pao2, paco2_chartoffset, paco2
	from pivoted_bg0
	

