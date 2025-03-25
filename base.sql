set search_path to mimiciii;
with base_v as (
select
 subject_id,
 hadm_id,
 icustay_id,
 gender,
 ethnicity,
 age,
 los_hospital,
 intime
from (
 select
 pa.subject_id,
 adm.hadm_id,
 ie.icustay_id,
 pa.gender,
 case when adm.ethnicity like '%WHITE%' then 'White' else 'Non-white' end as ethnicity,
 round(( cast(ie.intime as date) - cast(pa.dob as date))/365.2 )as age,
 round((date_part('epoch'::text, adm.deathtime - adm.admittime) / (60 * 60)::double precision)::numeric, 4) as survive,
 adm.admittime,
 adm.deathtime,
 adm.admission_location,
 ie.first_careunit,
 ie.intime,
 ie.outtime,
 round((date_part('epoch'::text, adm.dischtime - adm.admittime) / (60 * 60 * 24)::double precision)::numeric, 4) as los_hospital,
 row_number() over (partition by pa.subject_id, adm.hadm_id order by ie.intime) as admitMICU_order 
 from patients pa
 inner join admissions adm on pa.subject_id = adm.subject_id
 inner join icustays ie on pa.subject_id = ie.subject_id and adm.hadm_id = ie.hadm_id
 inner join diagnoses_icd icd on pa.subject_id = icd.subject_id and adm.hadm_id = icd.hadm_id
 where adm.admission_type = 'EMERGENCY'
   and ie.first_careunit = 'MICU'
   and round((date_part('epoch'::text, adm.deathtime - adm.admittime) / (60 * 60)::double precision)::numeric, 4) >= 48
   and icd.icd9_code in (select icd9_code from d_icd_diagnoses
 where short_title like '%sepsis%' or short_title like '%Sepsis%')
 order by pa.subject_id, adm.hadm_id, ie.icustay_id
)base 
where admitMICU_order = 1 and age >= 18 and age <= 89
),
weight_v as (
select
 b.subject_id,
 b.hadm_id,
 b.icustay_id,
 avg(ch.valuenum) weight
from base_v b inner join chartevents ch 
on b.subject_id = ch.subject_id and b.hadm_id = ch.hadm_id and b.icustay_id = ch.icustay_id
where ch.valuenum is not null and ch.itemid in (762, 763, 226512, 224639) 
 and ch.valuenum <> 0::double precision and ch.error is distinct from 1
group by b.subject_id, b.hadm_id, b.icustay_id
)

select
 b.*, w.weight
from base_v b left join weight_v w 
   on  b.subject_id = w.subject_id and b.hadm_id = w.hadm_id and b.icustay_id = w.icustay_id
where weight is not null
order by b.subject_id, b.hadm_id, b.icustay_id