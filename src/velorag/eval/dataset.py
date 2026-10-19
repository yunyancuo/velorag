from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalDocument:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class EvalSample:
    id: str
    query: str
    relevant_doc_ids: list[str]
    expected_keywords: list[str] = field(default_factory=list)
    should_refuse: bool = False
    category: str = "factual"


def get_builtin_campus_benchmark() -> tuple[list[EvalDocument], list[EvalSample]]:
    """Built-in bilingual Campus & AI Lab Knowledge Base benchmark dataset."""
    docs = [
        EvalDocument(
            doc_id="doc_grad_handbook",
            title="研究生学籍与培养管理手册 (2026版)",
            text=(
                "研究生培养方案与毕业要求提供完整的学分规范。硕士研究生培养体系要求总学分不少于32学分，"
                "其中学位必修课不少于18学分，跨学科选修课不少于4学分。研究生在申请学位论文答辩前，"
                "必须完成开题报告并通过中期考核。开题报告与学位论文答辩之间的时间间隔不得少于10个月。"
            ),
            metadata={"source": "研究生培养手册_2026.pdf", "department": "研究生院"},
        ),
        EvalDocument(
            doc_id="doc_gpu_cluster",
            title="AI开源实验室算力集群使用规范",
            text=(
                "AI开源实验室算力集群采用Slurm作业调度系统进行管理。单卡A100调试队列(debug)最长运行时间为4小时，"
                "多卡训练队列(train)最长运行时间为72小时。每位实验室正式成员默认拥有500GB高速NVMe存储配额。"
                "提交分布式训练任务需使用sbatch脚本并指定gpu资源数量。"
            ),
            metadata={"source": "AI实验室算力指南.md", "department": "AI开源实验室"},
        ),
        EvalDocument(
            doc_id="doc_med_ai_joint",
            title="医疗AI联合培养项目科研与数据合规制度",
            text=(
                "医疗AI联合培养项目依托附属医院与人工智能学院共建。所有医学影像数据（CT、MRI及病理切片）"
                "在进入训练流水线前必须完成DICOM匿名化脱敏处理。模型推理评测基于VeloRAG检索增强框架与知识图谱验证，"
                "严禁将含有患者隐私标识符的数据上传至公网第三方API。"
            ),
            metadata={"source": "医疗AI联培数据合规规范.docx", "department": "人工智能学院"},
        ),
        EvalDocument(
            doc_id="doc_reimbursement",
            title="学术会议差旅与开源项目资助报销办法",
            text=(
                "研究生参加CCF-A类或顶级国际学术会议发表第一作者论文，可申请最高15000元国际差旅资助。"
                "报销材料包含会议录用通知、注册费发票、登机牌及住宿水单，并需在返校后15个工作日内提交至财务系统。"
                "开源实验室立项项目可额外申请每月800元云算力与API调用补贴。"
            ),
            metadata={"source": "学术差旅与开源资助办法.pdf", "department": "财务处"},
        ),
        EvalDocument(
            doc_id="doc_velorag_arch",
            title="VeloRAG Architecture and Graph Retrieval Specification",
            text=(
                "VeloRAG implements hybrid retrieval combining dense vector search, multilingual BM25 sparse scoring, "
                "and NetworkX multi-hop knowledge graph traversal. VeloRAG uses Reciprocal Rank Fusion (RRF) with k=60 "
                "and enforces inline [Source N] citations with automatic evidence refusal when relevance falls below threshold."
            ),
            metadata={"source": "velorag_spec.md", "department": "OpenSourceLab"},
        ),
    ]

    samples = [
        EvalSample(
            id="q1_credits",
            query="硕士研究生毕业总学分和学位必修课分别要求不少于多少学分？",
            relevant_doc_ids=["doc_grad_handbook"],
            expected_keywords=["32", "18"],
            should_refuse=False,
            category="campus_policy",
        ),
        EvalSample(
            id="q2_gpu_slurm",
            query="AI开源实验室算力集群使用什么调度系统？train训练队列最长运行多少小时？",
            relevant_doc_ids=["doc_gpu_cluster"],
            expected_keywords=["Slurm", "72"],
            should_refuse=False,
            category="lab_infra",
        ),
        EvalSample(
            id="q3_med_dicom",
            query="医疗AI联合培养项目中，医学影像数据在训练前必须完成什么处理？",
            relevant_doc_ids=["doc_med_ai_joint"],
            expected_keywords=["DICOM", "匿名化"],
            should_refuse=False,
            category="medical_ai",
        ),
        EvalSample(
            id="q4_reimburse",
            query="参加CCF-A类国际会议最高可申请多少差旅资助？返校后多少个工作日内需要提交报销？",
            relevant_doc_ids=["doc_reimbursement"],
            expected_keywords=["15000", "15"],
            should_refuse=False,
            category="finance",
        ),
        EvalSample(
            id="q5_velorag_en",
            query="How does VeloRAG fuse dense vector search, BM25, and knowledge graph retrieval?",
            relevant_doc_ids=["doc_velorag_arch"],
            expected_keywords=["Reciprocal Rank Fusion", "RRF"],
            should_refuse=False,
            category="technical_en",
        ),
        EvalSample(
            id="q6_multihop_med_velorag",
            query="医疗AI联合培养项目所使用的VeloRAG框架采用什么融合算法（RRF）与脱敏规范？",
            relevant_doc_ids=["doc_med_ai_joint", "doc_velorag_arch"],
            expected_keywords=["DICOM", "RRF"],
            should_refuse=False,
            category="multihop_crosslingual",
        ),
        EvalSample(
            id="q7_unanswerable_canteen",
            query="请问紫金港校区第三食堂的夜宵麻辣烫几点关门？",
            relevant_doc_ids=[],
            expected_keywords=[],
            should_refuse=True,
            category="unanswerable_trap",
        ),
        EvalSample(
            id="q8_unanswerable_crypto",
            query="What is the exact Bitcoin price prediction for December 2029 according to the handbook?",
            relevant_doc_ids=[],
            expected_keywords=[],
            should_refuse=True,
            category="unanswerable_trap",
        ),
    ]
    return docs, samples
