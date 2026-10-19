# Residual retrieval audit record

## Initial audit — 01ac893 against 85a88f1 (verbatim)

1. **CONFIRMED — Float32 cancellation becomes a full-strength residual signal.**  
   File: `C:\Users\34684\velorag\src\velorag\retrieval\residual.py`, lines 18–20, 100–101, 137–138.  
   Excerpts: `"dtype=np.float32"`, `"values / np.maximum(..., 1e-8)"`, and `"residuals = unit_rows(self._vectors - projection * references)"`.  
   Exactly collinear embeddings should have zero residual. Float32 projection/subtraction can leave errors around \(10^{-7}\), exceeding the normalization floor; the code normalizes this numerical noise to approximately unit length. A read-only probe with three identical 384-dimensional vectors produced residual self-match scores of **0.9607, 0, and −0.0146** for different fixed input vectors. Their mathematical residuals were all zero. The same problem exists in query residual construction. Consequently, duplicate and nearly redundant passages can receive arbitrary residual scores that change candidate ordering and selected weights. Its incidence in the pending semantic run is unverified.

2. **CONFIRMED — Fusion truncates candidates in an order inconsistent with its emitted scores.**  
   Files: `C:\Users\34684\velorag\scripts\benchmark_residual.py`, line 113; traced dependency `C:\Users\34684\velorag\src\velorag\retrieval\engine.py`, lines 180, 188–193.  
   Claim: `"# Combine normalized RRF with best normalized signal score"`.  
   What runs: `"sorted(scores.items(), key=lambda x: x[1], reverse=True)"` orders by **RRF alone**, then `"combined_score = 0.65 * ... + 0.35 * ..."` creates a different score; the benchmark immediately takes `"[:depth]"`.  
   A read-only counterexample returned `[("a", 0.65), ("b", 0.6857), ("d", 0.6857), ...]`. Thus a smaller-score document can occupy the candidate boundary while a larger-score document is dropped. Lexical reranking later sorts using these combined scores, so its measured difference also includes correcting this ordering discrepancy. This is inherited baseline code, but the new empirical comparison executes it and selects the residual weight using its results.

3. **CONFIRMED — New residual modes lose persisted documents after restarting the application.**  
   File: `C:\Users\34684\velorag\src\velorag\pipeline.py`, lines 61, 146–149; exposed by `src\velorag\cli.py`, lines 33 and 47.  
   Excerpts: `"self._all_chunks: list[Chunk] = []"` and `"if self._all_chunks: index.fit(self._all_chunks, ...)"`.  
   Residual construction reads only documents ingested into the current object. The existing Chroma backend independently opens its persisted collection; those documents never populate `_all_chunks`. Therefore a fresh application or CLI process can have a searchable persisted dense collection while the new residual modes return no documents. This affects application demonstrations and any evaluation performed through that persisted pipeline. The standalone SciFact script explicitly fits the corpus and does not encounter this particular defect.

4. **CONFIRMED — The pilot cannot report retrieval quality together with query latency.**  
   File: `C:\Users\34684\velorag\scripts\benchmark_residual.py`, lines 116–129 and 194–200.  
   Excerpts: `"dense_scores = vectors @ qvector"`, `"residual_scores = ... index.scores(qvector)"`, and report fields `"index": index.stats()` / `"validation": summarize(...)`.  
   Query execution is never timed, and the report saves no per-mode query latency or throughput. Embedding duration and residual-index build duration are different quantities. Consequently, any observed quality change cannot be assessed against its query-time cost, contrary to the repository’s research convention. This matters because every evaluation call executes both dense and residual searches plus both hybrid and reranking paths.

Claims vs. code: The fusion comment describes a combined signal whose ordering is not used before truncation; the commit message makes no additional unsupported empirical claim.

Sample: The preparation manifest records 5,183 corpus lines, 1,109 query lines, and 919 qrel data rows. After the positive-score filter, qrel set collapse, random query selection, sparse positive-score filtering, source cutoffs, fused cutoff, and final metric cutoff: **N unverified** in run logs. Defaults request 40 development and 120 validation queries; these are requested counts, not verified completed-run counts. No undocumented corpus drop was found.

Merges: Membership checks reject missing relevant documents and missing judged queries; fitted chunk IDs must be unique, embedding counts must match, and baseline/residual document order is checked. A separate read-only inspection found unique corpus/query IDs and no duplicate qrel query–document pairs in the available inputs. No unmatched-observation or duplicate-key defect was found; `_merge` and id-period panels are inapplicable.

Variables: Traced normalized document/query embeddings, neighborhood references, projected residuals, convex scores, fusion scores, lexical scores, binary Recall@K and NDCG@K. Confirmed residual numerical instability and inconsistent fusion score ordering. Logs, deflation, and lag alignment are inapplicable.

Silent failures: Confirmed near-zero residual noise is silently amplified; confirmed residual searches silently use an empty current-process corpus after a persisted-index restart. No missing-to-zero coercion or silent embedding fallback was found in the standalone pilot.

Estimation: Development labels alone select the weight, with smaller-weight tie breaking; validation metrics are unweighted query means. Statistical clustering, fixed effects, and regression estimation are absent. Completed evaluation N and any empirical quality difference remain **N unverified**.

Could not check without the completed run data: how often the numerical and fusion defects affect actual SciFact candidate rankings and reported validation differences.

## Reaudit — 125e8a7 against 85a88f1 (verbatim)

Reaudit of `125e8a7` against `85a88f1` found two remaining material integration errors. The initial findings remain valid for `01ac893`.

1. **CONFIRMED — Sharded persistent snapshots duplicate every document and make residual construction fail.**  
   File: `C:\Users\34684\velorag\src\velorag\indexing\sharding.py`, lines 67–73.  
   Excerpt: `"for shard in self.shards: ... chunks.extend(shard_chunks)"`.  
   Traced construction: `pipeline.py:43` supplies `"vector_store_factory=lambda s: create_vector_store(s)"`; `sharding.py:29` passes identical settings to every factory call. Chroma therefore opens the same persistent directory and collection for every shard. The new snapshot concatenates these identical collections. `ResidualIndex.fit` then rejects repeated chunk IDs. A read-only reproduction returned snapshot IDs `["same", "same"]`, followed by `ValueError: Chunk IDs must be unique`. This breaks both residual modes when `num_shards > 1` uses Chroma. The standalone SciFact pilot does not use this path.

2. **CONFIRMED — Reloaded residual-hybrid and baseline-hybrid use different sparse corpora.**  
   File: `C:\Users\34684\velorag\src\velorag\pipeline.py`, lines 147–156; baseline initialization at lines 54–58 and 63.  
   Excerpts: `"chunks, embeddings = self.vector_store.snapshot()"`, `"retriever.set_chunks(chunks)"`, and `"self._all_chunks: list[Chunk] = []"`.  
   Snapshot reconstruction populates BM25 only for the residual retriever. The baseline retriever remains without a BM25 index after reopening persisted vectors. A read-only reproduction returned baseline hybrid source `"dense"` but residual hybrid source `"hybrid(local_residual+sparse)"` for the same saved document. Consequently, a comparison after restart attributes restoration of the sparse source to residual retrieval, even with residual weight zero. This affects pipeline evaluations; the SciFact script explicitly builds both methods’ BM25 corpus and avoids it.

Original defect status:

| Initial finding | Current verification |
|---|---|
| Float32 cancellation promoted to residual signal | **Fixed.** Across 100 fixed random vectors, three identical copies produced exactly zero residual keys and scores. |
| Fusion order inconsistent with emitted scores | **Fixed.** The original counterexample now returns scores in descending order before truncation. |
| Persisted corpus missing from residual modes | **Fixed for a single store.** Aligned stored embeddings now build the residual index; sharded and baseline-parity errors above remain. |
| Missing query latency | **Fixed at component level.** Every mode has per-query timing and summary p50/p95; the report explicitly distinguishes this from API latency and throughput. |

All **17 existing residual tests passed** under system Python. The project virtual environment lacks pytest; no dependencies or files were modified.

Claims vs. code: The repaired numerical, fusion-order, and component-timing behavior matches its comments; persisted reconstruction remains incomplete for sharded stores and paired baseline comparisons.

Sample: An independent execution of current `load_data` logged **5,183 corpus rows → 5,183 retained; 1,109 query rows; 919 qrel rows → 919 positive pairs → 809 judged query IDs → 160 selected → 40 development + 120 validation**. Completed benchmark candidate/final counts and metric N remain **N unverified** until the repaired run finishes.

Merges: Corpus/query uniqueness and qrel-pair duplication now fail visibly; membership and embedding alignment checks remain. Confirmed duplicate IDs arise when the new sharded snapshot concatenates aliased persistent collections.

Variables: Degenerate residuals, weight overrides, convex score construction, and emitted score order verified; no remaining defect found in the standalone pilot’s score or outcome construction.

Silent failures: Original numerical amplification fixed; reloaded baseline sparse retrieval still silently disappears while residual sparse retrieval is restored.

Estimation: Weight selection still uses development labels only; validation summaries remain unweighted query means. All mode timing keys were finite and nonnegative in a synthetic probe. Clusters, fixed effects, and regression weights are inapplicable.

Could not check without the completed repaired run data: whether its actual candidate counts, latency distribution, selected weight, and validation metrics match the recorded sample flow and corrected ranking behavior.

## Bounded repair verification — a2b5e70 (verbatim)

已限定核验 `a2b5e70` 的两处修复，未发现这两处仍有实质性错误。

- **分片重复快照：已修复。** 相同 ID、内容和向量合并为一条；冲突重复显式报错；搜索结果按 ID 去重。独立验证三个共享存储的分片，快照和搜索均只返回一条。
- **恢复后的 sparse corpus 不一致：已修复。** 分别以五种模式首次搜索，baseline hybrid 与 residual hybrid 均恢复 sparse 来源。另验证“首次搜索前追加写入”，两种检索器都保留已存文档及新增文档。

相关测试 **4 passed，14 deselected**；独立只读探针全部通过。未编辑文件，也未扩展审计范围。

## Author run disposition

The pre-audit process loaded old modules while embeddings were being prepared. Its result is invalid and excluded. Only checksum-verified BGE embeddings were reused. The corrected pilot ran after a2b5e70; the supplemental cross-encoder ran from 9af874a (the CLI argument recorded the symbolic value HEAD). Later argument guards do not alter the completed run. Both original result JSON files are retained with their source hashes.

## Completed results verification (verbatim)

发现一项待解释的复现差异，未发现两份验证结果存在实质性计算错误。

**SUSPECTED — 开发集四个非零权重 NDCG 汇总无法精确重现。**  
文件：`data/experiments/residual/scifact-pilot-corrected.json`，第 319、373、401、427 行。例：`"ndcg_at_10": 0.7139964879710433`。使用匹配的输入、缓存、源码及相同配置，多次重放得到：

| 权重 / 模式 | 保存值 | 重放值 |
|---|---:|---:|
| 0.1 / local_residual | 0.713996488 | 0.717269732 |
| 0.25 / local_residual | 0.709714367 | 0.710286213 |
| 0.25 / residual_hybrid_lexical_rerank | 0.679744564 | 0.683017807 |
| 0.5 / local_residual | 0.672693921 | 0.673460061 |

原因尚未确定。脚本第 217 行仅保存开发集汇总：`tuning.append({"weight": weight, "metrics": metrics})`，没有逐查询开发集排名，因而无法定位具体查询。**实际用于选权的 residual_hybrid 四项开发 NDCG 均精确重现，重新选择仍为 weight=0。**

其余核验结果：

- **N 正确：** 40 条开发查询、120 条验证查询及120条结果；CE 为同序验证集前40条，40条结果，开发与验证 ID 无交集。
- **指标正确：** 两份报告全部逐查询 Recall@10、NDCG@10 均可从 `top_ids` 和相关文档集合独立重算；全部质量均值及主报告延迟分位数一致。
- **候选公平：** 重建120条验证查询及CE40条查询，无排名或候选 recall 差异。CE每种方法使用自身原始50候选；共享 union 仅复用推理，未扩大任何方法的候选集合。保存分数重排得到的CE top10全部一致。
- **无 fallback：** supplemental 脚本调用真实 `TextCrossEncoder`，本地实现走 ONNX 推理；异常或无效分数直接失败，没有词法替代路径。
- **哈希正确：** 数据、embedding缓存、两种ONNX模型资产及主报告记录的源码哈希均匹配。CE脚本哈希精确匹配提交 **9af874a** 的Git blob；当前工作树已有后续修改。CE报告提交字段为字面量 `"HEAD"`，具体快照需依靠该哈希及9af874a确认。
- **报告限制：** 仅凭报告中的top10，无法独立重算 `candidate_recall_at_50`；CE的union分数也未记录各方法完整50候选归属。本次通过校验缓存和源码重建完整候选后，确认候选 recall 一致。

未修改文件。

## Recorded-environment development replay (verbatim)

已在新进程设置 `OPENBLAS_NUM_THREADS=4`、`OMP_NUM_THREADS=4` 核验，未编辑文件。

- 新报告的源码哈希、NumPy构建信息及四个拟合数组哈希全部匹配。
- 四组开发集trace各40条，查询顺序一致；逐查询top10重放、指标独立重算及汇总均无差异。
- 原先四个字段现均精确重现：`0.7139964879710433`、`0.7097143668930987`、`0.6797445635648733`、`0.6726939208444154`。

此前 **SUSPECTED** 作为历史审计记录保留；此次在报告记录的环境下未复现差异。不据此推断原因。
