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

inputevents_mv1 as (
select *, 
 case when rateuom = 'mL/min' then rate*60 else rate end new_rate
from inputevents_mv
),

f1 as (
select
 subject_id, hadm_id, icustay_id, stage1, sum(amount1) amount1
from (
 select
 b.subject_id,
 b.hadm_id,
 b.icustay_id,
 mv.amount amount1,
 case when mv.amount > 0 then 1 else 0 end as stage1
 from base_v b inner join inputevents_mv mv 
 on b.subject_id = mv.subject_id and b.hadm_id = mv.hadm_id and b.icustay_id = mv.icustay_id
 where mv.itemid in (select itemid from d_items where category = 'Fluids/Intake')
 and ( round((date_part('epoch', mv.starttime - b.intime) / (60 * 60)::double precision)::numeric, 4) <= 3
   and round((date_part('epoch', mv.endtime - b.intime) / (60 * 60)::double precision)::numeric, 4) <= 3 )
)t
where stage1 = 1
group by subject_id, hadm_id, icustay_id, stage1
),

f2 as (
select
 subject_id, hadm_id, icustay_id, stage2, sum(amount2) amount2
from (
 select
 b.subject_id,
 b.hadm_id,
 b.icustay_id,
 mv.amount amount2,
 case when mv.amount > 0 then 1 else 0 end as stage2
 from base_v b inner join inputevents_mv mv 
 on b.subject_id = mv.subject_id and b.hadm_id = mv.hadm_id and b.icustay_id = mv.icustay_id
 where mv.itemid in (select itemid from d_items where category = 'Fluids/Intake')
   and (round((date_part('epoch', mv.starttime - b.intime) / (60 * 60)::double precision)::numeric, 4) >= 3
    and round((date_part('epoch', mv.starttime - b.intime) / (60 * 60)::double precision)::numeric, 4) <= 24
    and round((date_part('epoch', mv.endtime - b.intime) / (60 * 60)::double precision)::numeric, 4) >= 3
    and round((date_part('epoch', mv.endtime - b.intime) / (60 * 60)::double precision)::numeric, 4) <= 24)
)t
where stage2 = 1
group by subject_id, hadm_id, icustay_id, stage2
),

ff1 as (
select
 subject_id, hadm_id, icustay_id, stage1, sum(amount1) amount1
from (
 select
  b.subject_id,
  b.hadm_id,
  b.icustay_id,
     case when round((date_part('epoch', mv.endtime - b.intime) / (60 * 60)::double precision)::numeric, 4) <= 3
          then mv.amount
          else round((date_part('epoch', (b.intime+interval '3 H') - mv.starttime) / (60 * 60)::double precision)::numeric, 4)*new_rate
     end as amount1,
   case when mv.amount > 0 then 1 else 0 end as stage1
 from base_v b inner join inputevents_mv1 mv 
 on b.subject_id = mv.subject_id and b.hadm_id = mv.hadm_id and b.icustay_id = mv.icustay_id
 where mv.itemid in (select itemid from d_items where category = 'Fluids/Intake')
   and round((date_part('epoch', mv.starttime - b.intime) / (60 * 60)::double precision)::numeric, 4) < 3
)t
where stage1 = 1
group by subject_id, hadm_id, icustay_id, stage1
),

ff2 as (
select
 subject_id, hadm_id, icustay_id, stage2, sum(amount2) amount2
from (
 select
  b.subject_id,
  b.hadm_id,
  b.icustay_id,
  case when ((round((date_part('epoch', mv.starttime - b.intime) / (60 * 60)::double precision)::numeric, 4) >= 3
               and round((date_part('epoch', mv.starttime - b.intime) / (60 * 60)::double precision)::numeric, 4) < 24)
               and (round((date_part('epoch', mv.endtime - b.intime) / (60 * 60)::double precision)::numeric, 4) >= 3
               and round((date_part('epoch', mv.endtime - b.intime) / (60 * 60)::double precision)::numeric, 4) < 24))
         then mv.amount
         when ((round((date_part('epoch', mv.starttime - b.intime) / (60 * 60)::double precision)::numeric, 4) >= 3
               and round((date_part('epoch', mv.starttime - b.intime) / (60 * 60)::double precision)::numeric, 4) < 24)
               and round((date_part('epoch', mv.endtime - b.intime) / (60 * 60)::double precision)::numeric, 4) > 24)
         then round((date_part('epoch', (b.intime+interval '24 H') - mv.starttime) / (60 * 60)::double precision)::numeric, 4)*new_rate
         when (round((date_part('epoch', mv.starttime - b.intime) / (60 * 60)::double precision)::numeric, 4) < 3
               and (round((date_part('epoch', mv.endtime - b.intime) / (60 * 60)::double precision)::numeric, 4) > 3
               and round((date_part('epoch', mv.endtime - b.intime) / (60 * 60)::double precision)::numeric, 4) <= 24))
         then round((date_part('epoch', mv.endtime - (b.intime+interval '3 H')) / (60 * 60)::double precision)::numeric, 4)*new_rate
         else 21*new_rate
    end as amount2,
  case when mv.amount > 0 then 1 else 0 end as stage2
  from base_v b inner join inputevents_mv1 mv 
  on b.subject_id = mv.subject_id and b.hadm_id = mv.hadm_id and b.icustay_id = mv.icustay_id
  where mv.itemid in (select itemid from d_items where category = 'Fluids/Intake')
     and ((round((date_part('epoch', mv.starttime - b.intime) / (60 * 60)::double precision)::numeric, 4) >= 3
              and round((date_part('epoch', mv.starttime - b.intime) / (60 * 60)::double precision)::numeric, 4) < 24)
          or (round((date_part('epoch', mv.endtime - b.intime) / (60 * 60)::double precision)::numeric, 4) > 3
             and round((date_part('epoch', mv.endtime - b.intime) / (60 * 60)::double precision)::numeric, 4) <= 24) 
          or (round((date_part('epoch', mv.starttime - b.intime) / (60 * 60)::double precision)::numeric, 4) <= 3
             and round((date_part('epoch', mv.endtime - b.intime) / (60 * 60)::double precision)::numeric, 4) >= 24))
)t
where stage2 = 1
group by subject_id, hadm_id, icustay_id, stage2
)

select
  coalesce(ff1.subject_id, ff2.subject_id) subject_id,
  coalesce(ff1.hadm_id, ff2.hadm_id) hadm_id,
  coalesce(ff1.icustay_id, ff2.icustay_id) icustay_id,
  ff1.amount1 fluid1,
  ff2.amount2 fluid2
from ff1 full join ff2 
                on ff1.subject_id = ff2.subject_id and ff1.hadm_id = ff2.hadm_id and ff1.icustay_id = ff2.icustay_id
order by coalesce(ff1.subject_id, ff2.subject_id),
       coalesce(ff1.hadm_id, ff2.hadm_id),
       coalesce(ff1.icustay_id, ff2.icustay_id)