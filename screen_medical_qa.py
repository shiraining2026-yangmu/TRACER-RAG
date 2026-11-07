#!/usr/bin/env python3
"""Conservative, reproducible subject/quality screening; never certify clinical truth.

Reads JSONL inside the original ZIP without extraction. Frozen, answer-blind
exam stems are used ONLY for overlap exclusion, never for topic/quality selection.
Does not build an index, call a model, rewrite answers, or read official keys.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
import zipfile

from run_dense_chunk_rag import ContaminationGuard, compact, file_hash, questions

VERSION = "medical-qa-subject-screen-v2"
TOPICS = {
    "中医基础与治则": "阴阳 五行 藏象 脏腑 精气 津液 气血 气机 升降出入 五脏 六腑 奇恒之腑 肝主疏泄 肝藏血 心主血脉 心藏神 脾主运化 肺主宣发 肺主肃降 肾藏精 命门 三焦 营气 卫气 宗气 元气 正气 邪气 六淫 七情 病机 同病异治 异病同治 辨证论治 扶正祛邪 治未病 标本缓急 正治 反治 因人制宜 因时制宜 因地制宜".split(),
    "中医诊断与辨证": "辨证 辨證 证候 证型 八纲 六经辨证 卫气营血 三焦辨证 望闻问切 四诊 望诊 闻诊 问诊 切诊 舌诊 脉诊 脉象 浮脉 沉脉 迟脉 数脉 滑脉 涩脉 弦脉 虚脉 实脉 气虚 血虚 阴虚 阳虚 气滞 血瘀 痰湿 湿热证 寒湿证 肝郁 脾虚 肾虚 肺热 胃热 心火 痰热 瘀血 阳亢 表证 里证 寒证 热证 虚证 实证 亡阴 亡阳".split(),
    "方剂": "方剂 君臣佐使 君药 臣药 佐药 使药 麻黄汤 桂枝汤 银翘散 桑菊饮 小青龙汤 大青龙汤 小柴胡汤 大柴胡汤 四君子汤 四物汤 八珍汤 十全大补汤 六味地黄 知柏地黄 杞菊地黄 金匮肾气 右归丸 左归丸 补中益气 归脾汤 生脉散 参苓白术 理中丸 小建中汤 吴茱萸汤 真武汤 四逆汤 四逆散 当归四逆 逍遥散 逍遥丸 龙胆泻肝 黄连解毒 清营汤 犀角地黄 白虎汤 玉女煎 大承气 小承气 调胃承气 温脾汤 麻子仁丸 二陈汤 温胆汤 半夏泻心 平胃散 藿香正气 五苓散 猪苓汤 六一散 三仁汤 八正散 小蓟饮子 保和丸 枳实导滞 越鞠丸 血府逐瘀 桃核承气 桂枝茯苓 天麻钩藤 镇肝熄风 羚角钩藤 酸枣仁汤 天王补心 安宫牛黄 苏合香丸 朱砂安神 瓜蒌薤白 百合固金 麦门冬汤 清燥救肺 止嗽散 定喘汤 独活寄生 双黄连".split(),
    "中药学": "中药学 性味 归经 四气 五味 升降浮沉 七情配伍 相须 相使 相畏 相杀 相恶 相反 十八反 十九畏 炮制 先煎 后下 包煎 烊化 冲服 人参 党参 黄芪 黄耆 白术 茯苓 甘草 熟地 生地 山茱萸 山药 泽泻 牡丹皮 当归 白芍 川芎 黄连 黄芩 黄柏 栀子 金银花 连翘 麻黄 桂枝 附子 干姜 肉桂 半夏 陈皮 枳实 枳壳 柴胡 葛根 升麻 防风 荆芥 薄荷 牛蒡子 桔梗 杏仁 瓜蒌 贝母 麦冬 天冬 玄参 石膏 知母 竹叶 薏苡仁 厚朴 苍术 砂仁 木香 香附 丹参 红花 桃仁 益母草 牛膝 杜仲 续断 桑寄生 酸枣仁 远志 龙骨 牡蛎 天麻 钩藤 地龙 全蝎 蜈蚣 乌头 马钱子 细辛 朱砂 雄黄 牵牛子 巴豆 甘遂 大戟 芫花 海藻 藜芦 五灵脂".split(),
    "针灸经络": "针灸学 经络 经脉 腧穴 穴位 奇经 任脉 督脉 带脉 冲脉 十二经 十五络 五输穴 原穴 络穴 郄穴 募穴 俞穴 下合穴 八会穴 八脉交会 交会穴 经穴 奇穴 阿是穴 针刺治疗 针刺穴位 艾灸疗法 得气 补泻 行针 留针 毫针 电针 穴位配伍 合谷 足三里 三阴交 内关 外关 太冲 太溪 肾俞 肝俞 脾俞 肺俞 心俞 中脘 关元 气海 神阙 百会 风池 风府 肩井 曲池 列缺 尺泽 鱼际 神门 大陵 曲泽 阳陵泉 阴陵泉 委中 承山 昆仑 涌泉 命门 大椎 膻中 中极 天枢 环跳 梁丘 伏兔 血海 公孙 丰隆 解溪 水沟 印堂 迎香 听宫 翳风 地仓 颊车 攒竹 睛明 四白".split(),
    "中医临床学科": "消渴 痹证 痿证 淋证 癃闭 不寐 心悸 怔忡 郁证 癫证 狂证 痫证 噎膈 反胃 噫气 呃逆 泄泻 痢疾 积聚 鼓胀 黄疸 胁痛 胸痹 肺痈 肺痨 肺胀 喘证 哮病 痰饮 水肿 血证 虚劳 内伤发热 疟疾 厥证 痉证 脱证 闭证 崩漏 经行 月经先期 月经后期 经间期 带下病 产后恶露 胎漏 胎动不安 小儿疳积 疳证 惊风 痄腮 丹痧 白喉 顿咳 乳痈 瘰疬 瘿病 疮疡 肠痈 痔疮 脱肛 白疕 蛇串疮 湿疮 伤筋 骨伤".split(),
}
# Ambiguous everyday strings are NOT topic anchors: e.g. 湿热敷, 针刺样疼痛,
# 主治医师, 病因 and 中医医院 do not establish TCM educational relevance.
TOPIC_RE = {topic: re.compile("|".join(map(re.escape, sorted(words, key=len, reverse=True))))
            for topic, words in TOPICS.items()}
EDUCATIONAL = re.compile(r"什么是|何谓|是指|指什么|有(?:什么|何)区别|的区别|有(?:什么|何)不同|"
                         r"的(?:作用|功效|组成|含义|特点|意义|原则|机制|位置|病机|治法)|"
                         r"由什么组成|包括哪些|有哪些|是什么|为什么|如何理解|怎么理解|在哪里|如何定位|怎么定位")
PERSONAL = re.compile(r"我(?:的|是|今|这|最|想|有|得|患|吃|喝|妈|爸|母|父|老|儿|女|现)|"
                     r"本人|宝宝|孩子|医生您好|医生你好|大夫您好|大夫你好|请医生|"
                     r"\d+岁|今年.{0,3}岁|患病|确诊|检查报告|检查结果|B超|核磁|磁共振|"
                     r"最近|昨天|前天|上个月|半个月|一个月了|几年了|几天了|手淫|"
                     r"我|自己|家人|小孩|医生|大夫|病情|是什么情况|这是什么症状|什么原因|"
                     r"早上|晚上|夜晚|左脚|右脚|左手|右手|一天|两个月|三天|几年|多久|"
                     r"检查|体质.*分|这个方子|这个药方|会出现|经常|怎么调理")
# These words can describe ordinary biomedical issues without TCM content.
AMBIGUOUS_CLINICAL = {"痔疮", "水肿", "黄疸", "脱肛", "噫气", "呃逆", "反胃", "经行", "骨伤", "伤筋"}
CONCRETE_EDUCATIONAL = re.compile(r"什么是|何谓|有(?:什么|何)区别|的区别|有(?:什么|何)不同|"
                                 r"的(?:作用|功效|组成|含义|特点|意义|原则|机制|位置|病机|治法)|"
                                 r"由什么组成|包括哪些|有哪些|是什么|为什么|如何理解|怎么理解|"
                                 r"在哪里|如何定位|怎么定位|怎么分别|如何区分|怎么区分|怎么分辨|如何分辨|如何区别")
PRESCRIPTIVE = re.compile(r"建议.{0,18}(?:吃|服|用|食|补|保暖|温热)|多食用|多吃|少吃|忌食|"
                          r"饮食调理|应该.{0,10}(?:吃|服|用)|可以.{0,10}(?:吃|服用|喝)|"
                          r"具体.{0,8}(?:方法|步骤)|(?:每日|每天|每次|一周).{0,8}[次服用]")
REFUSAL = re.compile(r"作为.{0,8}(?:助手|模型|AI)|我不是.{0,8}(?:医生|专业人士)|"
                     r"无法(?:提供|回答)|不能(?:提供医疗|为您提供)|不清楚您|没有足够的信息")
BOILERPLATE = re.compile(r"建议.{0,30}(?:就医|医院|医生|检查)|请.{0,15}(?:医生|就医)|"
                         r"多喝水|注意休息|保持.{0,12}(?:习惯|心情|饮食)|避免.{0,12}(?:劳累|熬夜)|"
                         r"适当运动|均衡饮食|规律作息|清淡饮食")
COMPOSITION = re.compile(r"(?:成分|组成|组方|药物).{0,16}(?:包括|包含|有|是|为|如下)|"
                         r"由.{2,120}(?:组成|制成)|(?:组成|配方)[：:]|分别是")
DOSE = re.compile(r"(?:\d+(?:\.\d+)?|[一二三四五六七八九十半]+)\s*(?:mg|ml|毫克|毫升|克|两|钱|片|粒|袋|丸|付|剂|滴|寸)(?![\u4e00-\u9fff]*(?:以上字段))", re.I)
PROCEDURE = re.compile(r"进针|刺入|针刺深度|留针|每次.{0,10}分钟|每天.{0,6}[次服]|每日.{0,6}[次服]")
OVERCLAIM = re.compile(r"保证.{0,8}(?:治好|治愈)|包治|彻底根治|百分之百|100%治|"
                       r"(?:完全没有|绝对没有)副作用|(?:可以|能够|能)根治")
CONTACT = re.compile(r"https?://|www\.|微信|加群|联系.{0,4}客服|\b1[3-9]\d{9}\b|\b\d{17}[\dXx]\b")
META = re.compile(r"忽略.{0,8}(?:指令|提示)|system\s*:|<\|(?:im_start|system)|标准答案[：:]|正确答案[：:]", re.I)


def normalized(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def anchors(text):
    return {topic: sorted(set(regex.findall(text))) for topic, regex in TOPIC_RE.items()
            if regex.search(text)}


def screen(question, answer):
    """Returns tier, reasons, topical signals. Tier is NOT a factual confidence."""
    if not isinstance(question, str) or not isinstance(answer, str):
        return "excluded", ["non_string_qa"], {}
    q, a = normalized(question), normalized(answer)
    if not q or not a:
        return "excluded", ["empty_qa"], {}
    if len(q) < 5 or len(a) < 35:
        return "excluded", ["too_short_for_reusable_knowledge"], {}
    if "\ufffd" in q + a or any(0xE000 <= ord(c) <= 0xF8FF for c in q + a):
        return "excluded", ["unresolved_character_encoding"], {}
    aq, aa = anchors(q), anchors(a)
    if not aq and not aa:
        return "excluded", ["no_specific_tcm_subject_anchor"], {}
    signals = {"question_anchors": aq, "answer_anchors": aa}
    if META.search(q + a):
        return "excluded", ["instruction_or_answer_key_artifact"], signals
    if CONTACT.search(q + a):
        return "excluded", ["contact_identifier_or_external_link"], signals
    if OVERCLAIM.search(a):
        return "excluded", ["absolute_efficacy_claim"], signals
    na = sum(map(len, aa.values()))
    all_terms = set(t for terms in (list(aq.values()) + list(aa.values())) for t in terms)
    if all_terms <= AMBIGUOUS_CLINICAL:
        return "excluded", ["biomedical_term_without_tcm_context"], signals
    if not aq and na < 2:
        return "excluded", ["incidental_tcm_mention_in_generic_answer"], signals
    substantive = BOILERPLATE.sub("", a)
    if not aa or (REFUSAL.search(a) and na < 3) or len(substantive) < 45:
        return "excluded", ["no_substantive_tcm_answer"], signals
    flags = []
    if not aq:
        flags.append("tcm_only_in_answer")
    if PERSONAL.search(q):
        flags.append("individual_patient_context")
    if not CONCRETE_EDUCATIONAL.search(q):
        flags.append("not_explicit_educational_question")
    if len(q) > 180:
        flags.append("long_consultation_question")
    if DOSE.search(q + " " + a):
        flags.append("dose_or_procedure_quantity_requires_reference")
    if PROCEDURE.search(a):
        flags.append("procedural_instruction_requires_reference")
    if PRESCRIPTIVE.search(a):
        flags.append("prescriptive_advice_requires_reference")
    if "方剂" in aa and COMPOSITION.search(a):
        flags.append("formula_composition_requires_reference")
    if re.search(r"(?:可以|可|能够|具有|用于|治疗|缓解|改善|疗效|禁忌|孕妇|哺乳|适合|适用|安全)", a) and (
            set(aa) & {"方剂", "中药学", "针灸经络"}):
        flags.append("treatment_or_drug_claim_requires_reference")
    if not a.endswith(("。", "！", "？", ".", "!", "?", "）", ")", "”", '"')):
        flags.append("possible_truncated_answer")
    if REFUSAL.search(a):
        flags.append("answer_contains_model_refusal")
    if na < 2:
        flags.append("limited_subject_specific_detail")
    # Concrete observed error pattern, not an exhaustive medical verifier.
    if "双黄连" in a and "黄连" in a.replace("双黄连", "") and COMPOSITION.search(a):
        return "excluded", ["observed_shuanghuanglian_component_conflict"], signals
    return ("review_required", flags, signals) if flags else ("priority_candidate", [], signals)


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def screen_archive(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    exams = questions(args.exam_questions)
    stems = [(r["question_id"], r["retrieval_query"]) for _, rows in exams for r in rows]
    guard = ContaminationGuard([s for _, s in stems])
    source_sha = file_hash(args.archive)
    source = {"archive": str(args.archive.resolve()), "archive_sha256": source_sha, "member": args.member,
              "provenance_status": "publisher_default_readme_no_record_level_references"}
    counts, reasons = Counter(), Counter()
    records, rejected_samples = [], defaultdict(list)
    q_answers = defaultdict(set)
    member_hash = hashlib.sha256()
    start = time.monotonic()
    with zipfile.ZipFile(args.archive) as archive, archive.open(args.member) as f, (
        out / "decisions.jsonl").open("w", encoding="utf-8") as audit:
        for line_no, line in enumerate(f, 1):
            member_hash.update(line)
            if not line.strip():
                continue
            counts["source_records"] += 1
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("record_not_object")
            except (ValueError, UnicodeError):
                row = {}
                tier, why, signals = "excluded", ["invalid_json_record"], {}
            else:
                tier, why, signals = screen(row.get("input"), row.get("output"))
            q, a = row.get("input"), row.get("output")
            if isinstance(q, str) and isinstance(a, str):
                nq, na = compact(normalized(q)), compact(normalized(a))
                if nq:
                    q_answers[sha(nq)].add(sha(na))
            record = None
            if tier != "excluded":
                match = guard.match(q) or guard.match(a)
                if match:
                    tier, why = "excluded", ["frozen_exam_stem_overlap"]
                    signals["excluded_overlap"] = {"question_id": stems[match[0]][0], "rule": match[1]}
                else:
                    record = {"record_id": "CMQA-" + sha(source_sha + ":" + str(line_no))[:20],
                              "source_line": line_no, "question": normalized(q), "answer": normalized(a),
                              "screen_tier": tier, "review_flags": why,
                              "topics": sorted(set(signals["question_anchors"]) | set(signals["answer_anchors"])),
                              "signals": signals, "source": source,
                              "factual_verification": "not_verified", "eligible_for_index": False}
                    records.append(record)
            counts["initial_" + tier] += 1
            for reason in why:
                reasons[reason] += 1
            audit.write(json.dumps({"line": line_no, "initial_tier": tier, "reasons": why,
                                    **({"overlap": signals["excluded_overlap"]} if "excluded_overlap" in signals else {})}, ensure_ascii=False) + "\n")
            if tier == "excluded" and len(rejected_samples[why[0]]) < 4:
                rejected_samples[why[0]].append({"source_line": line_no, "question": q, "answer": a,
                                                "reason": why[0]})
            if counts["source_records"] % 50000 == 0:
                print(json.dumps({"processed": counts["source_records"], "priority": counts["initial_priority_candidate"],
                                  "review": counts["initial_review_required"],
                                  "seconds": round(time.monotonic() - start, 1)}), flush=True)
    # Differing answers to an identical normalized question remain for review;
    # do not silently choose one potentially incorrect answer.
    seen_pairs, seen_answers = {}, {}
    duplicates = []
    kept = []
    inverted = defaultdict(set)
    prior_qgrams = []
    for record in sorted(records, key=lambda r: (r["screen_tier"] != "priority_candidate", r["source_line"])):
        q, a = compact(record["question"]), compact(record["answer"])
        pair = sha(q + "\0" + a)
        if pair in seen_pairs:
            duplicates.append({"record_id": record["record_id"], "source_line": record["source_line"],
                               "reason": "normalized_qa_duplicate", "duplicate_of": seen_pairs[pair]})
            continue
        seen_pairs[pair] = record["record_id"]
        if len(q_answers[sha(q)]) > 1:
            record["review_flags"].append("same_question_different_answers")
            record["screen_tier"] = "review_required"
        # Repeated answers are often boilerplate; retain only the first record
        # with the identical answer, even if its question differs.
        if len(a) >= 60 and sha(a) in seen_answers:
            duplicates.append({"record_id": record["record_id"], "source_line": record["source_line"],
                               "reason": "normalized_answer_duplicate", "duplicate_of": seen_answers[sha(a)]})
            continue
        seen_answers[sha(a)] = record["record_id"]
        # Only merge near-identical QA if BOTH question and answer agree strongly.
        # Similar questions with different answers must not be silently collapsed.
        qgrams = {q[j:j+3] for j in range(len(q)-2)}
        matches = Counter(i for g in qgrams for i in inverted.get(g, ()))
        duplicate_of = None
        for i, common in sorted(matches.items()):
            if common / max(1, len(qgrams) + len(prior_qgrams[i]) - common) < .85:
                continue
            old = kept[i]
            oldq = compact(old["question"])
            text, oldtext = q + "|" + a, oldq + "|" + compact(old["answer"])
            grams = {text[j:j+5] for j in range(len(text)-4)}
            oldgrams = {oldtext[j:j+5] for j in range(len(oldtext)-4)}
            if len(grams & oldgrams) / max(1, len(grams | oldgrams)) >= .92:
                duplicate_of = old["record_id"]
                break
        if duplicate_of:
            duplicates.append({"record_id": record["record_id"], "source_line": record["source_line"],
                               "reason": "near_duplicate_qa", "duplicate_of": duplicate_of})
            continue
        position = len(kept)
        for gram in qgrams:
            inverted[gram].add(position)
        prior_qgrams.append(qgrams)
        kept.append(record)
    priority = sorted([r for r in kept if r["screen_tier"] == "priority_candidate"], key=lambda r: r["source_line"])
    review = sorted([r for r in kept if r["screen_tier"] == "review_required"], key=lambda r: r["source_line"])
    # Separate generally phrased educational questions from historical consultations.
    # Neither pool is medically verified; no pool is automatically indexed.
    educational_review = [r for r in review if r["signals"]["question_anchors"]
                          and CONCRETE_EDUCATIONAL.search(r["question"])
                          and not PERSONAL.search(r["question"])
                          and len(r["question"]) <= 180]
    educational_ids = {r["record_id"] for r in educational_review}
    consultation_review = [r for r in review if r["record_id"] not in educational_ids]
    write_jsonl(out / "priority_candidates.jsonl", priority)
    write_jsonl(out / "review_required.jsonl", review)
    write_jsonl(out / "educational_review.jsonl", educational_review)
    write_jsonl(out / "consultation_review.jsonl", consultation_review)
    write_jsonl(out / "duplicates.jsonl", duplicates)
    write_jsonl(out / "excluded_examples.jsonl", [r for group in rejected_samples.values() for r in group])
    contract = {"version": VERSION, "script_sha256": file_hash(Path(__file__)), "source": source,
                "member_sha256": member_hash.hexdigest(), "topic_terms": TOPICS,
                "exam_stems": [{"path": str(p.resolve()), "sha256": file_hash(p), "count": len(rows)} for p, rows in exams],
                "selection": "fixed subject anchors and reusable educational question/answer features; no exam outcomes or keys",
                "overlap": "question AND answer: normalized stem containment >=12 or character-5gram Jaccard >=0.80",
                "dedup": "normalized QA; identical answer >=60 chars; QA 5gram >=0.92 AND Q 3gram >=0.85",
                "limitations": ["No record-level sources", "Rule-based screening does not verify factual correctness",
                                "Paraphrased exam contamination may remain", "Existing corpus cross-dedup has not been run",
                                "All outputs require reference verification before indexing or training"]}
    (out / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"counts": dict(counts), "priority_candidates": len(priority), "review_required": len(review),
               "educational_review": len(educational_review), "consultation_review": len(consultation_review),
               "duplicates_removed": len(duplicates), "duplicate_reasons": dict(Counter(r["reason"] for r in duplicates)),
               "initial_reason_counts": dict(reasons),
               "priority_topics_multilabel": dict(Counter(t for r in priority for t in r["topics"])),
               "review_flags": dict(Counter(t for r in review for t in r["review_flags"])),
               "source_questions_with_different_answers": sum(len(v) > 1 for v in q_answers.values()),
               "elapsed_seconds": round(time.monotonic() - start, 1)}
    assert sum(counts["initial_" + tier] for tier in ("excluded", "priority_candidate", "review_required")) == counts["source_records"]
    assert len(priority) + len(review) + len(duplicates) == len(records)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preview = ["# 优先候选全文复核", "", "这是规则筛出的待核验材料，不是医学正确性认证。没有使用标准答案。", ""]
    for r in priority:
        preview.extend([f"## {r['record_id']}（原文件第 {r['source_line']} 行）", "",
                        "主题：" + "、".join(r["topics"]), "", "问题：" + r["question"], "", "回答：" + r["answer"], ""])
    (out / "priority_review.md").write_text("\n".join(preview), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--member", default="chinese-medical/medicalQA.json")
    parser.add_argument("--exam-questions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    screen_archive(parser.parse_args())


if __name__ == "__main__":
    main()
