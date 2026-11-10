#!/usr/bin/env python3
"""Validate released supporting records and recompute descriptive summaries.

No model calls, private corpora, pandas, NumPy or SciPy are required.
Archived confidence intervals are preserved, not silently recalculated.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paired_statistics import exact_mcnemar

DATA = ROOT / "reproducibility/supporting_data"
MODELS = {
    "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
    "qwen3_8b": "Qwen3-8B",
    "internlm3_8b_instruct": "InternLM3-8B-Instruct",
    "meta_llama_3_1_8b_instruct": "Llama-3.1-8B-Instruct",
    "mistral_7b_instruct_v03": "Mistral-7B-Instruct-v0.3",
    "gemma2_9b_it": "Gemma-2-9B-it",
    "yi15_9b_chat": "Yi-1.5-9B-Chat",
    "qwen35_9b": "Qwen3.5-9B",
}
ARMS = ("dense", "ter", "evicr")
EXCLUDED = "QB-2022-TCM-03-080"
RELATIONS = {"diagnostic_support", "explanatory_support", "treatment_or_operation_support"}
PAIRED_FIELDS = "model year comparison n baseline_correct candidate_correct both_correct baseline_correct_candidate_wrong baseline_wrong_candidate_correct both_wrong gain_minus_loss_questions paired_difference_percentage_points independent_question_mcnemar_n independent_question_mcnemar_gains independent_question_mcnemar_losses independent_question_mcnemar_exact_two_sided_p independent_question_mcnemar_holm_adjusted_p all_question_mcnemar_unclustered_sensitivity_only_p question_bootstrap_ci95_percentage_points cluster_bootstrap_ci95_percentage_points cluster_count".split()
STRICT_FIELDS = "model arm year correct scored accuracy_percent y0_gains y0_losses y0_delta_accuracy_pp y0_ci95_low y0_ci95_high y0_mcnemar_singletons_p y1_gains y1_losses y1_delta_accuracy_pp y1_ci95_low y1_ci95_high y1_mcnemar_singletons_p answer_prompt_tokens answer_completion_tokens critic_prompt_tokens critic_completion_tokens".split()
SCHEMAS = {
    "archived_exam_correctness.csv": ["application_id", "year", "unit", "shared_option_group"] + [f"{m}__{a}" for m in MODELS for a in ARMS],
    "archived_paired_statistics_primary.csv": PAIRED_FIELDS,
    "archived_paired_statistics_sensitivity.csv": PAIRED_FIELDS,
    "structure_assessments.csv": "application_id arm status evidence_support historical_case_misuse key_condition_omission unsupported_expansion candidate_sha256".split(),
    "open_qa_ai_scores.csv": "row_id model correctness completeness safety composite_mean_0_to_2 judge_a_preference judge_b_preference".split(),
    "strict_ablation_aggregates.csv": STRICT_FIELDS,
    "evaluation_query_membership.csv": "application_id split relation_types".split(),
    "critic_seed_build_membership.csv": "sample_id application_id split relation_types".split(),
    "archived_no_rag_vs_dense.csv": "application_id year no_rag_correct dense_correct".split(),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def read(name):
    with (DATA / name).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == SCHEMAS[name], "Unexpected schema: " + name)
        rows = list(reader)
    require(all(set(row) == set(SCHEMAS[name]) and None not in row.values() for row in rows), "Malformed CSV: " + name)
    return rows


def close(actual, expected, tolerance=1e-10):
    require(math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= tolerance,
            f"Numeric mismatch: {actual} versus {expected}")


def binary(value):
    require(value in {"0", "1"}, "Nonbinary correctness")
    return int(value)


def validate_exam(matrix, table, exclude):
    results = []
    require(len(table) == 72, "Expected 72 archived paired comparisons")
    require(len({(r['model'],r['year'],r['comparison']) for r in table}) == 72, "Duplicate paired comparison")
    for record in table:
        model = next(m for m, label in MODELS.items() if label == record["model"])
        before, after = record["comparison"].split("_to_")
        require(before in ARMS and after in ARMS, "Unknown arm")
        rows = [r for r in matrix if (not exclude or r["application_id"] != EXCLUDED)
                and (record["year"] == "combined" or r["year"] == record["year"])]
        pairs = Counter((int(r[f"{model}__{before}"]), int(r[f"{model}__{after}"])) for r in rows)
        counts = {"n":len(rows), "baseline_correct":pairs[1,0]+pairs[1,1],
                  "candidate_correct":pairs[0,1]+pairs[1,1], "both_correct":pairs[1,1],
                  "baseline_correct_candidate_wrong":pairs[1,0],
                  "baseline_wrong_candidate_correct":pairs[0,1], "both_wrong":pairs[0,0],
                  "gain_minus_loss_questions":pairs[0,1]-pairs[1,0]}
        for key, value in counts.items():
            require(int(record[key]) == value, "Archived paired count mismatch: " + key)
        close(record["paired_difference_percentage_points"],100*(pairs[0,1]-pairs[1,0])/len(rows),0.000051)
        close(record["all_question_mcnemar_unclustered_sensitivity_only_p"],exact_mcnemar(pairs[0,1],pairs[1,0]))
        singleton = [r for r in rows if not r["shared_option_group"]]
        sp = Counter((int(r[f"{model}__{before}"]), int(r[f"{model}__{after}"])) for r in singleton)
        require(int(record["independent_question_mcnemar_n"]) == len(singleton), "Singleton count mismatch")
        require(int(record["independent_question_mcnemar_gains"]) == sp[0,1] and int(record["independent_question_mcnemar_losses"]) == sp[1,0], "Singleton transition mismatch")
        close(record["independent_question_mcnemar_exact_two_sided_p"],exact_mcnemar(sp[0,1],sp[1,0]))
        clusters = {(r["year"],r["unit"],"shared:"+r["shared_option_group"] if r["shared_option_group"] else r["application_id"]) for r in rows}
        require(int(record["cluster_count"]) == len(clusters), "Cluster count mismatch")
        for field in ["question_bootstrap_ci95_percentage_points","cluster_bootstrap_ci95_percentage_points"]:
            interval = json.loads(record[field])
            require(len(interval) == 2 and -100 <= interval[0] <= interval[1] <= 100, "Invalid archived interval")
        results.append({"model":model,"year":record["year"],"comparison":record["comparison"],**counts})
    return results


def validate():
    provenance = load("PROVENANCE.json")
    for item in provenance["artifacts"]:
        name = item["release_path"]
        require(Path(name).name == name, "Unsafe provenance path")
        require(hashlib.sha256((DATA/name).read_bytes()).hexdigest() == item["release_sha256"], "Export hash mismatch: " + name)
    require({p.name for p in DATA.glob("*.csv")} == set(SCHEMAS), "Unreviewed or missing CSV")
    matrix = read("archived_exam_correctness.csv")
    require(len(matrix) == len({r["application_id"] for r in matrix}) == 1138, "Exam IDs/count")
    require(Counter(r["year"] for r in matrix) == {"2021":554,"2022":584}, "Exam year counts")
    for row in matrix:
        require(re.fullmatch(r"QB-202[12]-TCM-0[1-4]-\d{3}",row["application_id"]), "Exam ID format")
        require(row["unit"] in {"第一","第二","第三","第四"}, "Unit enum")
        require(re.fullmatch(r"(?:\d+-\d+)?",row["shared_option_group"]), "Group ID format")
        for m in MODELS:
            for a in ARMS: binary(row[f"{m}__{a}"])
    summaries={}
    for name,exclude in [("primary",True),("sensitivity",False)]:
        summaries[name] = validate_exam(matrix,read(f"archived_paired_statistics_{name}.csv"),exclude)
    totals = {m:{a:sum(int(r[f"{m}__{a}"]) for r in matrix) for a in ARMS} for m in MODELS}
    require(totals["qwen35_9b"] == {"dense":929,"ter":930,"evicr":966}, "Revision archive changed")

    structure = read("structure_assessments.csv")
    require(len(structure) == len({(r['application_id'],r['arm']) for r in structure}) == 222, "Structure pairing")
    require(len({r['application_id'] for r in structure}) == 111, "Structure question count")
    stats={}
    protocol=load("structure_protocol.json")["automated_assessment"]["metrics"]
    for arm in ["ter_full","ter_flat"]:
        rows=[r for r in structure if r["arm"] == arm]
        require(len(rows) == 111, "Structure arm count")
        for row in rows:
            require(re.fullmatch(r"A-AEK-\d{7}",row["application_id"]) and row["status"] == "valid", "Structure ID/status")
            require(re.fullmatch(r"[a-f0-9]{64}",row["candidate_sha256"]), "Candidate hash")
            require(row["evidence_support"] in {"supported","partial","unsupported"}, "Support enum")
            require(all(row[k] in {"yes","no"} for k in ["historical_case_misuse","key_condition_omission","unsupported_expansion"]), "Assessment enum")
        counts={"strict_evidence_support":sum(r["evidence_support"] == "supported" for r in rows),
                **{k:sum(r[k] == "yes" for r in rows) for k in ["historical_case_misuse","key_condition_omission","unsupported_expansion"]}}
        for k,v in counts.items(): close(v/len(rows),protocol[k][arm+"_rate"])
        stats[arm]={"n":111,"counts":counts}

    openqa=read("open_qa_ai_scores.csv"); grouped=defaultdict(list)
    require(len(openqa) == 600, "Open QA rows")
    for r in openqa:
        require(re.fullmatch(r"\d+",r["row_id"]), "Open QA ID")
        require(r["model"] in {"strict-qwen35-generator","strict-mistral-generator"}, "Open QA model")
        for k in ["correctness","completeness","safety","composite_mean_0_to_2"]:
            require(math.isfinite(float(r[k])) and 0 <= float(r[k]) <= 2, "Score range")
        close(float(r["composite_mean_0_to_2"]),sum(float(r[k]) for k in ["correctness","completeness","safety"])/3)
        for k in ["judge_a_preference","judge_b_preference"]: require(r[k] in {"A","B","tie","neither"}, "Preference enum")
        grouped[r["row_id"]].append(r)
    require(len(grouped) == 300 and all(len(v) == len({r['model'] for r in v}) == 2 for v in grouped.values()), "Open QA pairs")
    for rows in grouped.values():
        for k in ["judge_a_preference","judge_b_preference"]: require(len({r[k] for r in rows}) == 1, "Inconsistent pair preference")
    means={m:{k:sum(float(r[k]) for r in openqa if r['model']==m)/300 for k in ['correctness','completeness','safety']} for m in sorted({r['model'] for r in openqa})}
    for m, scores in means.items():
        for k,v in scores.items(): close(v,load('open_qa_ai_protocol.json')['mean_dimension_scores_0_to_2_averaged_across_judges'][m][k],0.000051)

    strict=read("strict_ablation_aggregates.csv")
    require(len(strict) == len({(r['model'],r['arm'],r['year']) for r in strict}) == 24, "Strict aggregate count")
    for row in strict:
        require(row['model'] in {'qwen35','mistral'} and row['arm'] in {'dense','ter_y0','tuned_critic','evicr_tuned'}, "Strict arm/model")
        require(row['year'] in {'2021','2022','combined'}, "Strict year")
        for k,v in row.items():
            if k not in {'model','arm','year'} and v: require(math.isfinite(float(v)), "Nonfinite aggregate")
        require(int(row['scored']) == {'2021':554,'2022':583,'combined':1137}[row['year']], "Strict denominator")
        require(0 <= int(row['correct']) <= int(row['scored']), "Strict numerator")
        close(row['accuracy_percent'],100*int(row['correct'])/int(row['scored']),0.000051)
    for model in ['qwen35','mistral']:
        for arm in ['dense','ter_y0','tuned_critic','evicr_tuned']:
            items=[r for r in strict if r['model']==model and r['arm']==arm]
            require(sum(int(r['correct']) for r in items if r['year']!='combined') == next(int(r['correct']) for r in items if r['year']=='combined'), "Strict combined count")

    members=read('evaluation_query_membership.csv'); seeds=read('critic_seed_build_membership.csv')
    require(len(members) == len({r['application_id'] for r in members}) == 345, "Query membership")
    require(Counter(r['split'] for r in members) == {'build':119,'dev':115,'test':111}, "Query splits")
    require(len(seeds) == len({r['sample_id'] for r in seeds}) == 3000, "Seed membership")
    for r in members + seeds:
        require(re.fullmatch(r'A-AEK-\d{7}',r['application_id']), "Membership ID")
        require(set(r['relation_types'].split('|')) <= RELATIONS, "Membership relation enum")
    require(all(r['split']=='build' for r in seeds), "Seed Build boundary")
    require(not ({r['application_id'] for r in seeds} & {r['application_id'] for r in members if r['split']!='build'}), "Build seed/evaluation overlap")
    baseline=read('archived_no_rag_vs_dense.csv')
    require(len(baseline) == len({r['application_id'] for r in baseline}) == 1138, "Historical baseline rows")
    require({(r['application_id'],r['year']) for r in baseline} == {(r['application_id'],r['year']) for r in matrix}, "Historical baseline IDs")
    baseline_counts={k:sum(binary(r[k]) for r in baseline) for k in ['no_rag_correct','dense_correct']}
    require(baseline_counts == {'no_rag_correct':916,'dense_correct':908}, "Historical baseline totals")
    return matrix, {"version":"supporting-data-descriptive-summary-v1", "scope":"Archived results; conditional routing and AI-rating caveats apply. No new model inference.",
                    "exam_all_1138_correct":totals, "exam_paired_counts":summaries,
                    "structure_assessment_counts":stats, "open_qa_ai_mean_scores":means,
                    "historical_baseline_correct":baseline_counts,
                    "membership_counts":{"queries":len(members),"build_seeds":len(seeds)}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-summary',action='store_true',help='Write deterministic descriptive summary after reviewing data')
    parser.add_argument('--paired-output',type=Path,help='Export correctness-only pairs for paired_statistics.py')
    parser.add_argument('--model',choices=MODELS,default='qwen35_9b')
    parser.add_argument('--before',choices=ARMS,default='ter')
    parser.add_argument('--after',choices=ARMS,default='evicr')
    parser.add_argument('--all-items',action='store_true',help='Retain known overlap for 1138-item sensitivity; default excludes it')
    args=parser.parse_args()
    matrix, summary=validate()
    if args.write_summary:
        (DATA/'numeric_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    else:
        require(load('numeric_summary.json') == summary, 'Descriptive summary mismatch')
    if args.paired_output:
        require(not args.paired_output.exists(), 'Paired output already exists')
        args.paired_output.parent.mkdir(parents=True,exist_ok=True)
        with args.paired_output.open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['id','year','before_correct','after_correct','cluster_id'],lineterminator='\n')
            writer.writeheader()
            for r in matrix:
                if not args.all_items and r['application_id'] == EXCLUDED: continue
                writer.writerow({'id':r['application_id'],'year':r['year'],
                    'before_correct':r[f'{args.model}__{args.before}'], 'after_correct':r[f'{args.model}__{args.after}'],
                    'cluster_id':f"{r['year']}|{r['unit']}|shared:{r['shared_option_group']}" if r['shared_option_group'] else ''})
    print('PASS: 9 CSV schemas, source-export hashes, 144 paired count/test comparisons, structure/AI-score summaries and split boundaries.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
