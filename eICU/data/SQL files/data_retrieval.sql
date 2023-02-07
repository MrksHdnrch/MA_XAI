WITH distinct_patient_ids as (

	-- Only keep the first ICU stay per hospital stay and only one hospital stay per patient
	-- There is no systematic method for chronologically ordering different hospital stays 
	-- for the same patient within the same year
	-- => order by year and take the first in the table

	select patientunitstayid
	from 
		(select*,
		 row_number() over(partition by uniquepid order by hospitaldischargeyear, patienthealthsystemstayid, unitvisitnumber) as row_number
		 from eicu_crd.patient) rnr
	where row_number = 1 and unitvisitnumber = 1   -- delete patients with wrong unitvisitnumbers

)
	
	-- one stay for each of the 132969 distinct patients
 
 

, ids0 as (

	-- REMOVE
	-- age under 18 and larger than 89
	-- end of life care
	-- Distinction based on: https://github.com/MIT-LCP/eicu-code/pull/46/files
	-- and https://github.com/nus-mornin-lab/oxygenation_kc/blob/master/data-extraction/eICU/eicu_final_patient_results.sql


	select distinct(a1.patientunitstayid)
	from
	(
		((select i.patientunitstayid
		 from distinct_patient_ids as i 
		 inner join eicu_crd.careplangeneral as c 
			on i.patientunitstayid = c.patientunitstayid
		 inner join eicu_crd.patient p
			on i.patientunitstayid = p.patientunitstayid)

		 except

		(select i.patientunitstayid
		 from distinct_patient_ids as i 
		 inner join eicu_crd.careplangeneral as c 
			on i.patientunitstayid = c.patientunitstayid
		 inner join eicu_crd.patient p
			on i.patientunitstayid = p.patientunitstayid
		 where cplitemvalue in ('Do not resuscitate', 'No CPR', 'No intubation', 'Comfort measures only',
								'No augmentation of care','End of life')
			or age in ('> 89','','0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17')))

	  except 

	   -- Remove all patients from 'end-of-life' table 
	  (select patientunitstayid
	   from eicu_crd.careplaneol)
	) as a1

)





, patientvalues as (
	-- gender 0:male and 1:female
	-- unitdischargestatus  0: Alive and 1:Expired

	-- from patient table
	select patientunitstayid, age, (unitdischargeoffset / 1440) as icu_los
		, case
			when gender = 'Male' then 0 else 1 end as gender
		, case when unitdischargestatus = 'Expired' then 1 else 0 end as unitdischargestatus
	from eicu_crd.patient
	
)


, specialty_info as (
	-- Surgery: 0
	-- Other: 1
	-- Medical: 2
	
	select patientunitstayid, max(specialty_cat) as specialty_cat
	from (
		-- distinction based on https://en.wikipedia.org/wiki/Medical_specialty
		select patientunitstayid, 
		case
			when specialty in ('surgery-general',
						'surgery-cardiac',
						'surgery-neuro',
						 'surgery-trauma',
						 'surgery-vascular',
						 'surgery-orthopedic',
						 'surgery-critical care',
						 'surgery-plastic',
						 'surgery-transplant',
						 'surgery-otolaryngology head & neck',
						 'surgery-oral',
						 'surgery-pediatric',
						 'urology',
						 'otolaryngology',
						 'orthopedics',
						 'ophthalmology',
						 'oncology',
						 'obstetrics/gynecology'
						  ) then 0 
			when specialty in ('', 'ethics','anesthesiology', 'anesthesiology/CCM', 'emergency medicine',
							  'family practice','hospitalist','nurse','nurse practitioner','other','pain management','unknown'
							  ) then 1
			else 2 end as specialty_cat
		from eicu_crd.careplancareprovider ) as sp1
	group by patientunitstayid


)

, datatable0 as (
	
	select p.patientunitstayid, gender, age, unitdischargestatus, icu_los
		   , vent_start, oxygen_therapy_type,  supp_oxygen
	from patientvalues p
	inner join ids0 i
	on i.patientunitstayid = p.patientunitstayid
	left join oxygen_info o
	on p.patientunitstayid = o.icustay_id

)


, datatable1 as (
	
	-- Join all information together ************************

	select d0.patientunitstayid, gender, cast(age as numeric) as age, unitdischargestatus, icu_los
		    , vent_start, oxygen_therapy_type,  supp_oxygen
			, pao2_chartoffset, pao2,paco2_chartoffset, paco2
			, gcs_chartoffset , gcs
			, glucose_chartoffset, glucose, bicarb_chartoffset, bicarbonate
			, heartrate_chartoffset, heartrate, resprate_chartoffset, respiratoryrate, o2sat_chartoffset, spo2, temp_chartoffset, temperature
			, bp_offset, sbp, dbp, map
			, heartrate / sbp as shockindex
			, specialty_cat
		
	from datatable0	d0
	left join pivoted_lab l
	on d0.patientunitstayid = l.patientunitstayid
	left join pivoted_vital v
	on d0.patientunitstayid = v.patientunitstayid
	left join pivoted_bg bg
	on d0.patientunitstayid = bg.patientunitstayid
	left join pivoted_gcs gcs
	on d0.patientunitstayid = gcs.patientunitstayid
	left join specialty_info s
	on d0.patientunitstayid = s.patientunitstayid
	left join pivoted_bp bp
	on d0.patientunitstayid = bp.patientunitstayid
	

)


, allowedids as (

	-- remove ICU stays without full set of HR, SBP,DBP, MAP, RR and Temp
	-- and patients with >2 missing points of the rest of the variables
	select patientunitstayid as allowedids
	from datatable1
	where   heartrate is not NULL
			and sbp is not NULL
			and dbp is not NULL
			and map is not NULL
			and respiratoryrate is not NULL
			and temperature is not NULL

	except

	select patientunitstayid 
	from ( 
		select patientunitstayid as patientunitstayid , count(*) as null_values
		from
			(select patientunitstayid, spo2, gcs, bicarbonate, glucose, paco2, pao2, shockindex
			from datatable1 ) as t
		cross join jsonb_each_text(to_jsonb(t))
		where value is null
		group by patientunitstayid) as t1
	where null_values > 2

)

, datatable2 as (

	select *
	from datatable1 d1
	inner join allowedids a
	on d1.patientunitstayid = a.allowedids
		
	
)


, datatable3 as (
	
	-- Merge Vassopressor and SOFA Score information to existing datatable
	select d2.*
	, vasopressor_offset, vasopressor
	, sofatotal
	from datatable2 d2
	left join vasopressor_info v
	on d2.patientunitstayid = v.patientunitstayid
	left join sofa_info s
	on d2.patientunitstayid = s.patientunitstayid
	

)

, datatable4 as (
	-- create intubated dummy and fill empty specialty cells with unspecified
	select patientunitstayid, gender, age, unitdischargestatus, icu_los
		    , vent_start, oxygen_therapy_type,  supp_oxygen
			, pao2, paco2
			, gcs
			, glucose, bicarbonate
			, heartrate, respiratoryrate, spo2, temperature
			, sbp, dbp, map
			, heartrate / sbp as shockindex
			, vasopressor
			, sofatotal
		    , case
		  	  when specialty_cat is NULL then 1 else specialty_cat end as specialty
			, case
			  when (oxygen_therapy_type = 4 and vent_start > 0 and vent_start < 1440) then 1 else 0 end as intubated
	from datatable3
	)

select d4.*
into forpython1709_v3
from datatable4 d4
where oxygen_therapy_type != 2 and not (oxygen_therapy_type = 4 and vent_start < 0);  -- drop oxygen_therapy_type = 2 (either invasive or noninvasive => missleading) 
																					  -- and patients that have been intubated before ICU admission
-- ********************************************************
select *
from forpython1709_v3











 