set search_path to mimiciii;

with admitMICU_order_view as (
select subject_id, hadm_id, icustay_id, intime, 
row_number() over (partition by subject_id order by intime) as admitMICU_order 
from icustays
where first_careunit = 'MICU'
),

startMICU_order_view as (
select subject_id, icustay_id, weight
from (
 select subject_id, icustay_id, starttime, patientweight weight,
    row_number() over (partition by subject_id, icustay_id order by starttime) as startMICU_order
from inputevents_mv
) t1
where t1.startMICU_order=1
),
age_weight_view as (
select
 pat.subject_id,
 round((date_part('epoch'::text, adm_order.intime - pat.dob) / ((60 * 60 * 24)::numeric * 365.242)::double precision)::numeric, 4) as admMICU_age,
 input_order.weight
from patients pat
left join admitMICU_order_view adm_order on pat.subject_id = adm_order.subject_id
left join startMICU_order_view input_order on adm_order.subject_id = input_order.subject_id
  and adm_order.icustay_id = input_order.icustay_id
where adm_order.admitMICU_order = 1
),
base_v as (
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
),  

static_table as (
select
 distinct pat.subject_id,
 adm.hadm_id,
 icu.icustay_id,
 pat.gender,
 adm.ethnicity,
 age_weight_v.admMICU_age age,
 age_weight_v.weight,
 coalesce (round((date_part('epoch'::text, adm.deathtime - adm.admittime) / ((60 * 60)::numeric)::double precision)::numeric, 4), 48) as adm_survive
from patients as pat
left join admissions as adm on pat.subject_id = adm.subject_id
left join age_weight_view as age_weight_v on pat.subject_id = age_weight_v.subject_id
left join icustays as icu on pat.subject_id = icu.subject_id
where adm.diagnosis = 'SEPSIS'
  and adm.admission_type = 'EMERGENCY'
  and age_weight_v.admMICU_age >= 18
  and icu.first_careunit = 'MICU'
),
wt AS (
         SELECT ie_1.icustay_id,
            avg(
                CASE
                    WHEN c.itemid = ANY (ARRAY[762, 763, 3723, 3580, 226512]) THEN c.valuenum
                    WHEN c.itemid = 3581 THEN c.valuenum * 0.45359237::double precision
                    WHEN c.itemid = 3582 THEN c.valuenum * 0.0283495231::double precision
                    ELSE NULL::double precision
                END) AS weight
           FROM mimiciii.icustays ie_1
             LEFT JOIN mimiciii.chartevents c ON ie_1.icustay_id = c.icustay_id
          WHERE c.valuenum IS NOT NULL AND (c.itemid = ANY (ARRAY[762, 763, 3723, 3580, 3581, 3582, 226512])) AND c.valuenum <> 0::double precision AND c.charttime >= (ie_1.intime - '1 day'::interval day) AND c.charttime <= (ie_1.intime + '1 day'::interval day) AND c.error IS DISTINCT FROM 1
          GROUP BY ie_1.icustay_id
        ), echo2 AS (
         SELECT ie_1.icustay_id,
            avg(echo.weight * 0.45359237) AS weight
           FROM mimiciii.icustays ie_1
             LEFT JOIN mimiciii.echodata echo ON ie_1.hadm_id = echo.hadm_id AND echo.charttime > (ie_1.intime - '7 days'::interval day) AND echo.charttime < (ie_1.intime + '1 day'::interval day)
          GROUP BY ie_1.icustay_id
        ), vaso_cv AS (
         SELECT ie_1.icustay_id,
            max(
                CASE
                    WHEN cv.itemid = 30047 THEN cv.rate / COALESCE(wt.weight, ec.weight::double precision)
                    WHEN cv.itemid = 30120 THEN cv.rate
                    ELSE NULL::double precision
                END) AS rate_norepinephrine,
            max(
                CASE
                    WHEN cv.itemid = 30044 THEN cv.rate / COALESCE(wt.weight, ec.weight::double precision)
                    WHEN cv.itemid = ANY (ARRAY[30119, 30309]) THEN cv.rate
                    ELSE NULL::double precision
                END) AS rate_epinephrine,
            max(
                CASE
                    WHEN cv.itemid = ANY (ARRAY[30043, 30307]) THEN cv.rate
                    ELSE NULL::double precision
                END) AS rate_dopamine,
            max(
                CASE
                    WHEN cv.itemid = ANY (ARRAY[30042, 30306]) THEN cv.rate
                    ELSE NULL::double precision
                END) AS rate_dobutamine
           FROM mimiciii.icustays ie_1
             JOIN mimiciii.inputevents_cv cv ON ie_1.icustay_id = cv.icustay_id AND cv.charttime >= ie_1.intime AND cv.charttime <= (ie_1.intime + '1 day'::interval day)
             LEFT JOIN wt ON ie_1.icustay_id = wt.icustay_id
             LEFT JOIN echo2 ec ON ie_1.icustay_id = ec.icustay_id
          WHERE (cv.itemid = ANY (ARRAY[30047, 30120, 30044, 30119, 30309, 30043, 30307, 30042, 30306])) AND cv.rate IS NOT NULL
          GROUP BY ie_1.icustay_id
        ), vaso_mv AS (
         SELECT ie_1.icustay_id,
            max(
                CASE
                    WHEN mv.itemid = 221906 THEN mv.rate
                    ELSE NULL::double precision
                END) AS rate_norepinephrine,
            max(
                CASE
                    WHEN mv.itemid = 221289 THEN mv.rate
                    ELSE NULL::double precision
                END) AS rate_epinephrine,
            max(
                CASE
                    WHEN mv.itemid = 221662 THEN mv.rate
                    ELSE NULL::double precision
                END) AS rate_dopamine,
            max(
                CASE
                    WHEN mv.itemid = 221653 THEN mv.rate
                    ELSE NULL::double precision
                END) AS rate_dobutamine
           FROM mimiciii.icustays ie_1
             JOIN mimiciii.inputevents_mv mv ON ie_1.icustay_id = mv.icustay_id AND mv.starttime >= ie_1.intime AND mv.starttime <= (ie_1.intime + '1 day'::interval day)
          WHERE (mv.itemid = ANY (ARRAY[221906, 221289, 221662, 221653])) AND mv.statusdescription::text <> 'Rewritten'::text
          GROUP BY ie_1.icustay_id
        ), pafi1 AS (
         SELECT bg.icustay_id,
            bg.charttime,
            bg.pao2fio2,
                CASE
                    WHEN vd.icustay_id IS NOT NULL THEN 1
                    ELSE 0
                END AS isvent
           FROM mimiciii.bloodgasfirstdayarterial bg
             LEFT JOIN mimiciii.ventdurations vd ON bg.icustay_id = vd.icustay_id AND bg.charttime >= vd.starttime AND bg.charttime <= vd.endtime
          ORDER BY bg.icustay_id, bg.charttime
        ), pafi2 AS (
         SELECT pafi1.icustay_id,
            min(
                CASE
                    WHEN pafi1.isvent = 0 THEN pafi1.pao2fio2
                    ELSE NULL::double precision
                END) AS pao2fio2_novent_min,
            min(
                CASE
                    WHEN pafi1.isvent = 1 THEN pafi1.pao2fio2
                    ELSE NULL::double precision
                END) AS pao2fio2_vent_min
           FROM pafi1
          GROUP BY pafi1.icustay_id
        ), scorecomp AS (
         SELECT ie_1.icustay_id,
            v.meanbp_min,
            COALESCE(cv.rate_norepinephrine, mv.rate_norepinephrine) AS rate_norepinephrine,
            COALESCE(cv.rate_epinephrine, mv.rate_epinephrine) AS rate_epinephrine,
            COALESCE(cv.rate_dopamine, mv.rate_dopamine) AS rate_dopamine,
            COALESCE(cv.rate_dobutamine, mv.rate_dobutamine) AS rate_dobutamine,
            l.creatinine_max,
            l.bilirubin_max,
            l.platelet_min,
            pf.pao2fio2_novent_min,
            pf.pao2fio2_vent_min,
            uo.urineoutput,
            gcs.mingcs
           FROM mimiciii.icustays ie_1
             LEFT JOIN vaso_cv cv ON ie_1.icustay_id = cv.icustay_id
             LEFT JOIN vaso_mv mv ON ie_1.icustay_id = mv.icustay_id
             LEFT JOIN pafi2 pf ON ie_1.icustay_id = pf.icustay_id
             LEFT JOIN mimiciii.vitalsfirstday v ON ie_1.icustay_id = v.icustay_id
             LEFT JOIN mimiciii.labsfirstday l ON ie_1.icustay_id = l.icustay_id
             LEFT JOIN mimiciii.uofirstday uo ON ie_1.icustay_id = uo.icustay_id
             LEFT JOIN mimiciii.gcsfirstday gcs ON ie_1.icustay_id = gcs.icustay_id
        ), scorecalc AS (
         SELECT scorecomp.icustay_id,
                CASE
                    WHEN scorecomp.pao2fio2_vent_min < 100::double precision THEN 4
                    WHEN scorecomp.pao2fio2_vent_min < 200::double precision THEN 3
                    WHEN scorecomp.pao2fio2_novent_min < 300::double precision THEN 2
                    WHEN scorecomp.pao2fio2_novent_min < 400::double precision THEN 1
                    WHEN COALESCE(scorecomp.pao2fio2_vent_min, scorecomp.pao2fio2_novent_min) IS NULL THEN NULL::integer
                    ELSE 0
                END AS respiration,
                CASE
                    WHEN scorecomp.platelet_min < 20::double precision THEN 4
                    WHEN scorecomp.platelet_min < 50::double precision THEN 3
                    WHEN scorecomp.platelet_min < 100::double precision THEN 2
                    WHEN scorecomp.platelet_min < 150::double precision THEN 1
                    WHEN scorecomp.platelet_min IS NULL THEN NULL::integer
                    ELSE 0
                END AS coagulation,
                CASE
                    WHEN scorecomp.bilirubin_max >= 12.0::double precision THEN 4
                    WHEN scorecomp.bilirubin_max >= 6.0::double precision THEN 3
                    WHEN scorecomp.bilirubin_max >= 2.0::double precision THEN 2
                    WHEN scorecomp.bilirubin_max >= 1.2::double precision THEN 1
                    WHEN scorecomp.bilirubin_max IS NULL THEN NULL::integer
                    ELSE 0
                END AS liver,
                CASE
                    WHEN scorecomp.rate_dopamine > 15::double precision OR scorecomp.rate_epinephrine > 0.1::double precision OR scorecomp.rate_norepinephrine > 0.1::double precision THEN 4
                    WHEN scorecomp.rate_dopamine > 5::double precision OR scorecomp.rate_epinephrine <= 0.1::double precision OR scorecomp.rate_norepinephrine <= 0.1::double precision THEN 3
                    WHEN scorecomp.rate_dopamine > 0::double precision OR scorecomp.rate_dobutamine > 0::double precision THEN 2
                    WHEN scorecomp.meanbp_min < 70::double precision THEN 1
                    WHEN COALESCE(scorecomp.meanbp_min, scorecomp.rate_dopamine, scorecomp.rate_dobutamine, scorecomp.rate_epinephrine, scorecomp.rate_norepinephrine) IS NULL THEN NULL::integer
                    ELSE 0
                END AS cardiovascular,
                CASE
                    WHEN scorecomp.mingcs >= 13::double precision AND scorecomp.mingcs <= 14::double precision THEN 1
                    WHEN scorecomp.mingcs >= 10::double precision AND scorecomp.mingcs <= 12::double precision THEN 2
                    WHEN scorecomp.mingcs >= 6::double precision AND scorecomp.mingcs <= 9::double precision THEN 3
                    WHEN scorecomp.mingcs < 6::double precision THEN 4
                    WHEN scorecomp.mingcs IS NULL THEN NULL::integer
                    ELSE 0
                END AS cns,
                CASE
                    WHEN scorecomp.creatinine_max >= 5.0::double precision THEN 4
                    WHEN scorecomp.urineoutput < 200::double precision THEN 4
                    WHEN scorecomp.creatinine_max >= 3.5::double precision AND scorecomp.creatinine_max < 5.0::double precision THEN 3
                    WHEN scorecomp.urineoutput < 500::double precision THEN 3
                    WHEN scorecomp.creatinine_max >= 2.0::double precision AND scorecomp.creatinine_max < 3.5::double precision THEN 2
                    WHEN scorecomp.creatinine_max >= 1.2::double precision AND scorecomp.creatinine_max < 2.0::double precision THEN 1
                    WHEN COALESCE(scorecomp.urineoutput, scorecomp.creatinine_max) IS NULL THEN NULL::integer
                    ELSE 0
                END AS renal
           FROM scorecomp
        )


select
    w.subject_id,
 	w.hadm_id,
    w.icustay_id,
 	t3.sofa
from weight_v w
left join (
 SELECT ie.subject_id,
 ie.hadm_id,
 ie.icustay_id,
 COALESCE(s.respiration, 0) + COALESCE(s.coagulation, 0) + COALESCE(s.liver, 0) + COALESCE(s.cardiovascular, 0) + COALESCE(s.cns, 0) + COALESCE(s.renal, 0) AS sofa
 FROM icustays ie
  LEFT JOIN scorecalc s ON ie.icustay_id = s.icustay_id
 ORDER BY ie.subject_id, ie.hadm_id, ie.icustay_id
)t3 on w.subject_id = t3.subject_id and w.hadm_id = t3.hadm_id and w.icustay_id = t3.icustay_id
where w.weight is not null
order by subject_id, hadm_id, icustay_id

