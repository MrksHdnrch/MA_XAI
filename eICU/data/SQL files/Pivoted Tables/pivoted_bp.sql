-- Extract blood pressure variables:

drop table if exists pivoted_bp cascade;
create table pivoted_bp as (
 
select 
	v1.patientunitstayid, minvalue as bp_offset
	, avg(case when noninvasivesystolic >= 25 and noninvasivesystolic <= 250 then noninvasivesystolic else null end) as sbp
	, avg(case when noninvasivediastolic >= 1 and noninvasivediastolic <= 200 then noninvasivediastolic else null end) as dbp
	, avg(case when noninvasivemean >= 1 and noninvasivemean <= 250 then noninvasivemean else null end) as map
from
	(select distinct patientunitstayid
			,min(ABS(observationoffset)) * case when
	  		(min(case when observationoffset > 0 then observationoffset end) > abs(max(case when observationoffset < 0 then observationoffset end))
			 or 
			 (min(case when observationoffset > 0 then observationoffset end) is NULL))
		 		THEN -1 ELSE 1 END as minvalue -- extract minimum absolut chartoffset
			from eicu_crd.vitalaperiodic
			group by patientunitstayid) as v1
inner join eicu_crd.vitalaperiodic v
on v.patientunitstayid = v1.patientunitstayid
where abs(observationoffset) < 120
group by v1.patientunitstayid, bp_offset
)


