# Progress — 01/11/2026

## Bước tiếp theo

Review thủ công diagnosis, ưu tiên **97 QASPER selection-loss** và các regression
so với RRF pool50, rồi chốt một ablation chọn evidence với pool/model/ngân sách
cố định. Không chạy thêm retrieval/model/GPU hoặc mở reporting trong đợt này.
Đọc [báo cáo offline](../analysis/baseline_diagnosis_v1/REPORT.md),
[mẫu review](../analysis/baseline_diagnosis_v1/MANUAL_REVIEW.md) và
[hướng dẫn tiếp tục](NEXT_STEPS.md). Tài liệu này là nguồn theo dõi tiến độ chính.

Thí nghiệm đề xuất (chưa chạy): **QASPER selector có phạt trùng lặp và xét ngữ
cảnh lân cận**, so với top10 cross-encoder hiện tại, trên cùng union BM25+dense
top50, model SHA và ngân sách evidence/token. Review trước để kiểm tra liệu hai
tín hiệu này có phù hợp hay không; metric hiện tại không chứng minh nguyên nhân
là trùng lặp hoặc thiếu ngữ cảnh. Log toàn bộ candidate scores và evidence IDs
cho ablation; artifact hiện chỉ giữ reranker top10, không đủ để dựng lại score
của mọi candidate. Nếu cần suy luận model để lấy các scores này, đó là công việc
tương lai cần được cho phép riêng, không nằm trong phân tích offline này.
Chọn trước một protocol/ablation, không sweep theo diagnosis, giữ reporting kín.

## Phân biệt trạng thái

| Hạng mục | Code đã triển khai | Thực nghiệm/artifact đã có | Đã kiểm tra trong đợt này | Chưa làm |
|---|---|---|---|---|
| Dữ liệu và manifest | Prepare/validate; manifest chia theo paper/claim | Processed dev và manifests seed42 | Hash queries/documents, manifest, query coverage và group isolation | Không đổi manifest; chưa kiểm tra QA cuối |
| BM25/MiniLM | Runner full split, CPU/CUDA, cache | Dev top10 và top50, 1005 QASPER / 300 SciFact predictions | Report/data/source hashes, diagnosis metrics và denominator | Không chạy lại; embedding mạnh hơn chưa có |
| Hybrid RRF | Fusion prediction-only | RRF k=60 top10 từ nguồn top10 và từ nguồn top50 | Hash nguồn fusion, diagnosis scores, paired comparisons | Không coi hai độ sâu pool là cùng protocol |
| Cross-encoder | Pinned snapshot loader; GPU notebook resume A–G | Tesla T4 run, 1005/300 predictions, metrics diagnosis/calibration, backup | ZIP/manifest/config/source/report hashes; diagnosis scores/denominators | Chưa có full-union logits hoặc fine-tuning |
| Phân tích baseline | Offline script và fixtures | Markdown/CSV, paired bootstrap, danh sách review | Đã chạy cả hai dataset từ artifact local | Manual review 120 ví dụ chưa thực hiện |
| Selector/reader/controller | Chưa triển khai phương pháp mới | Chưa có thực nghiệm | Không tuyên bố cải thiện Answer F1/hallucination | Selector mới, reader/generation, oracle reader, adaptive controller |
| Đánh giá cuối/release | Evaluator có khả năng giữ reporting | Reporting/test chưa mở trong đợt này | Không parse/chấm gold reporting | Freeze phương pháp, held-out evaluation, demo |

`VERIFIED_COMPLETE` xác nhận hoàn tất và tính toàn vẹn của artifact, **không phải
bằng chứng chất lượng vượt baseline**. Bảng stage trong RESEARCH_PLAN là roadmap;
trạng thái hoàn thành phải đối chiếu ở đây.

## Repo và provenance

- Nhánh local kiểm tra: `feat/dev-retrieval-benchmark`.
- HEAD lúc bắt đầu: `83a5ee39cdb839400f040dd503fcbaff48c11576`
  (`fix: make reranker notebook resumable`). Đây không phải commit tạo mọi baseline.
- Trước khi sửa: `.gitignore` đã modified, `artifacts/` chưa tracked. Đã giữ nguyên
  các thay đổi đó; không reset/clean, không commit/push, không tải model/network.
- Source run reranker: `1d3ab1035545a867be1c649be17ad22565303e1f`.
- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`, SHA
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`, batch16, max-length512,
  float32, raw logit, top10, source depth50, CUDA Tesla T4.
- Dense: `sentence-transformers/all-MiniLM-L6-v2`, SHA
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, max-seq-length256,
  normalized float32 embeddings. BM25 k1=1.5, b=0.75, epsilon=0.25.
- Baseline Git provenance có giới hạn: BM25 ghi commit
  `455270a3eaf0cba82a0d207ed6e30ce129733301`, `dirty=true`; dense Git commit
  không được ghi (`null`); RRF không có Git commit field. Không gán commit source
  reranker cho các lần chạy này. Runner hashes của BM25/dense và RRF khớp bytes
  source trong archive; config/version/hardware có trong report/provenance.

| Source file trong archive | SHA256 |
|---|---|
| `scripts/run_retrieval_benchmark.py` | `7a0b0d6c7c59d3b83be1ca219c2c6aa9a7d2b6d7a9724a6474ab9b91f96675fe` |
| `scripts/fuse_retrieval_rrf.py` | `d1a65fdae7186035eeb6d77053f4df55a51edb94277a038beaa7ec4f34075504` |
| `scripts/rerank_retrieval.py` | `690f002ec546e38b6aeda582a94d145d1fd3184ff2984da77c203282658bba2f` |

## Artifact và hash

Archive local:
`artifacts/artifacts_VERIFIED_COMPLETE_20261031T170355811993Z_1bcc0097.zip`.
Sidecar cùng tên `.sha256.json`; SHA256 đã kiểm tra lại:
`a2aa11be3c93861739d79ebceb3219b3dc50d8b95d1b06ac93aaeb6011051bc0`.

Bản Drive của phiên:
`MyDrive/retrieval_benchmark/reranker_runs/20261031T164343Z_05af25c1/`.
Workspace lịch sử: `/content/reranker_20261031T164343Z_05af25c1`.
Đợt này chỉ đọc artifact local, không xác nhận lại trạng thái runtime/Drive.

Config `reranker_run_config.json` trong ZIP, SHA256:
`b162a087f864e03897c127a069b8e98551b3cc1a1d3e63803f17e2fa40ea31c7`.

| Artifact | QASPER SHA256 | SciFact SHA256 |
|---|---|---|
| Manifest `configs/dev_<dataset>_seed42.json` | `92a030e816fc4b74ab5dc4918b80dc20f059cf4d96641b71431b4b044aadf7bf` | `ff21870b4ba27ed9980752e566d0f8696b1b11d01cc9224f51eacd47aa3964f0` |
| Reranker `<dataset>_retrieval.json` trong ZIP, `evaluation/reports/dev_cross_encoder_v1/` | `bd9211644c4a193b79cbb64852ad7efb0669b3723137dd407c021372ff0b415b` | `ca8b79114781108d787836b29e78c8e76596134ba257ec193d503d2130e48bfd` |

Baseline local nằm trong `evaluation/reports/`: `dev_bm25_v1`,
`dev_dense_gpu_v1`, `dev_rrf_top10_v1`, `dev_rrf_pool50_top10_v1`.
Mỗi dataset có retrieval, metrics development, diagnosis. Source top50 được đọc
từ archive theo config và kiểm tra SHA256 với `rerank_sources`/`fusion_sources`.

[provenance.json](../analysis/baseline_diagnosis_v1/provenance.json) ghi SHA256
mọi input local/ZIP member, source/model config, metadata Git/hardware và
denominators. [output_hashes.json](../analysis/baseline_diagnosis_v1/output_hashes.json)
ghi SHA256 từng đầu ra, gồm chính script hash trong provenance. Đây là dữ liệu
máy đọc để đối chiếu chi tiết, không chỉ dựa vào tên ZIP hoặc nhãn hoàn thành.

## Kết quả diagnosis đã kiểm tra

Chỉ diagnosis được chấm từng query. Full-gold JSONL được hash như byte opaque
để kiểm tra toàn vẹn; không decode/parse gold calibration/reporting. Diagnosis
export chứa gold được đối chiếu giữa các method, evidence payload và scorer.
Predictions toàn dev chỉ dùng validation coverage/protocol, không xuất lỗi reporting.

| Dataset | Diagnosis selected / groups | Scored / skipped | Calibration selected / scored / skipped |
|---|---:|---:|---:|
| QASPER | 416 / 113 paper | 341 / 75 | 206 / 165 / 41 |
| SciFact | 120 / 120 claim | 71 / 49 | 60 / 39 / 21 |

Skip QASPER: `no_eligible_text_annotation`; SciFact: `no_annotated_evidence`.
Không tính skipped là retrieval miss, không loại âm thầm khỏi denominator.

| Dataset | Thiếu bộ evidence trong pool | Đủ trong pool nhưng top10 không đủ | Thành công top10 | Pool chứa đủ ít nhất một bộ |
|---|---:|---:|---:|---:|
| QASPER (341 scored) | 11 (3.23%) | 97 (28.45%) | 233 (68.33%) | 330 (96.77%) |
| SciFact (71 scored) | 9 (12.68%) | 3 (4.23%) | 59 (83.10%) | 62 (87.32%) |

Trong pool-complete, top10 thất bại 97/330 QASPER, 3/62 SciFact. Không có query
pool-complete nào cần hơn 10 normalized evidence units trong phân tích này.
Pool oracle có dùng gold diagnosis, không dùng được như selector thực tế.
Các bucket là quan sát set-containment; chưa gán nguyên nhân semantic.

Semantics giữ nguyên evaluator: QASPER đủ **một annotation text-only hợp lệ**,
normalized duplicate paragraph trong cùng paper tương đương nhưng vẫn chiếm rank;
MRR dùng union. SciFact recall dùng union câu của rationales, complete chỉ cần
**một rationale**. Không gộp scores hai dataset hoặc gọi đây là official scoring.

### Paired với RRF pool50 trên cùng pool nguồn

| Dataset | Metric | RRF pool50 | Reranker | Delta | Exploratory 95% CI |
|---|---|---:|---:|---:|---|
| QASPER | Recall@10 | 0.6698 | 0.7508 | +0.0810 | [+0.0381, +0.1230] |
| QASPER | Complete evidence@10 | 0.6041 | 0.6833 | +0.0792 | [+0.0330, +0.1256] |
| QASPER | MRR@10 | 0.3964 | 0.4525 | +0.0561 | [+0.0097, +0.1020] |
| SciFact | Recall@10 | 0.5942 | 0.6852 | +0.0910 | [+0.0332, +0.1567] |
| SciFact | Complete rationale@10 | 0.7465 | 0.8310 | +0.0845 | [+0.0267, +0.1558] |
| SciFact | MRR@10 | 0.4930 | 0.6108 | +0.1178 | [+0.0399, +0.1993] |

Chi tiết paired với BM25, dense và cả hai RRF nằm trong
[paired_diagnosis.csv](../analysis/baseline_diagnosis_v1/paired_diagnosis.csv).
QASPER reranker có 44 win / 280 tie / 17 loss complete-hit so với RRF pool50.
SciFact complete-hit so với dense có CI [-0.0154, +0.1343], chứa 0.
Không kết luận thắng mọi baseline ở mọi metric.

Bootstrap paired percentile 95%, 10.000 draw, seed42; QASPER resample toàn bộ
diagnosis paper và macro theo scored query với multiplicity, không average paper
means. SciFact resample claim. Group chỉ chứa skipped vẫn được resample; draw
denominator0 được loại và đếm (chi tiết CSV). CI diagnosis là **exploratory**,
không hiệu chỉnh multiple comparisons, không phải held-out confirmation.

### Calibration: chỉ số tổng hợp đã có

| Dataset | Metric | Dense | RRF top10 | RRF pool50 | Reranker |
|---|---|---:|---:|---:|---:|
| QASPER | Recall@10 | 0.5943 | 0.6399 | 0.6704 | 0.7492 |
| QASPER | Complete@10 | 0.5091 | 0.5636 | 0.5818 | 0.6727 |
| SciFact | Recall@10 | 0.5107 | 0.5620 | 0.4615 | 0.5214 |
| SciFact | Complete@10 | 0.6667 | 0.7436 | 0.5385 | 0.6410 |

SciFact calibration không cho thấy reranker thắng dense/RRF top10 về complete
hit; không tính CI hoặc phân tích lỗi query calibration. Chỉ 39 scored claim,
và calibration không thay thế final holdout sau tuning. Không suy ra nguyên nhân
phân phối/semantic từ sự khác biệt aggregate này.

## Output, kiểm thử và việc còn lại

`analysis/baseline_diagnosis_v1/` gồm `REPORT.md`, `aggregate_metrics.csv`,
`paired_diagnosis.csv`, `pool_diagnosis.csv`, `diagnosis_queries.csv`,
`MANUAL_REVIEW.md`, `provenance.json`, `output_hashes.json`.
Danh sách đầy đủ gồm 416 QASPER + 120 SciFact diagnosis query (có skipped);
mẫu manual review 60 query/dataset, seed42, phân tầng bucket, chưa được gán cause.

Đã chạy 5 fixture tests: duplicate/alternative annotation, pool vs selection,
skip/SciFact semantics, paper bootstrap query macro, denominator và archive-hash/
gold-reporting parsing guards. Đã chạy phân tích offline cả hai dataset thành công.
Không chạy retrieval/model/GPU, không tạo lại manifest, không mở reporting.

Còn lại: review ví dụ, kiểm tra giả thuyết selector, chốt một ablation và budget;
sau đó mới cân nhắc thực nghiệm model/selector mới. Chưa có kết luận Answer F1,
reader correctness, hallucination, novelty hoặc cải thiện final test. Không so
latency chéo hardware; rerank/fusion timings không bao gồm source retrieval.
