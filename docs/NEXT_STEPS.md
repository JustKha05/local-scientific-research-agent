# Bước tiếp theo: phân tích baseline trước phương pháp mới

Nguồn trạng thái chính: [PROGRESS.md](PROGRESS.md). Giữ nhánh
`feat/dev-retrieval-benchmark`; không quay về `feat/dense-retrieval` hoặc smoke.

## 1. Review kết quả đã có

Đọc [báo cáo offline](../analysis/baseline_diagnosis_v1/REPORT.md),
`paired_diagnosis.csv`, `pool_diagnosis.csv` và `provenance.json` trong cùng thư mục.
Phân biệt thiếu bộ evidence trong pool với có đủ trong pool nhưng top10 không giữ đủ.
`success` chỉ có nghĩa đủ ít nhất một annotation/rationale hợp lệ, không chứng minh
câu trả lời đúng hoặc citation được hỗ trợ. `VERIFIED_COMPLETE` không chứng minh
model tốt hơn. Không gộp điểm QASPER và SciFact.

## 2. Xác minh review QASPER diagnosis

Đọc [docs/MANUAL_REVIEW.md](MANUAL_REVIEW.md) và các packet liên kết trong
`analysis/qasper_review_v1/cases/`: question, gold hợp lệ theo alternative,
top10 reranker/RRF pool50, toàn bộ pool text/source ranks/positions.
Mẫu: 20 selection_loss + 10 regression complete-hit + 10 success, **37 unique
query / 31 paper**, chồng nhóm3, seed42. Xem `selection.json` và `observations.csv`.
Nhận xét trợ lý và facts tự động đã tách riêng; human labels **0/37, PENDING**.
Ghi reviewer/date, observation và evidence ID/text, counterexample và nhãn xác minh;
không tự suy lỗi semantic/truncation/reader từ metric hoặc annotation containment.
Không dùng mẫu phân tầng để ước lượng tỷ lệ lỗi quần thể.

0/37 có exact duplicate/high lexical overlap top10 theo ngưỡng đã chốt. 11/27
failure có gold thiếu nằm cạnh hit đã chọn;8/27 có alternative hoàn tất khi thêm
neighbor không giới hạn. Đây là gold-assisted structural oracle, không có token
cap/replacement và không phải gain selector. Đặc biệt xem các case annotation
ambiguity và success đủ một alternative dù chưa đủ union.

Tái lập packet khi cần (chỉ output-dir mới, không ghi đè v1):

```powershell
.\.venv\Scripts\python.exe -m unittest scripts.test_prepare_qasper_review scripts.test_analyze_baseline_diagnosis -v
.\.venv\Scripts\python.exe -m scripts.prepare_qasper_review --archive artifacts/artifacts_VERIFIED_COMPLETE_20261031T170355811993Z_1bcc0097.zip --sha256-file artifacts/artifacts_VERIFIED_COMPLETE_20261031T170355811993Z_1bcc0097.sha256.json --output-dir analysis/qasper_review_v2 --seed 42
```

Mẫu120 query cũ của baseline_v1 vẫn là artifact tham khảo, không có human labels;
không thay thế review37 theo yêu cầu hiện tại.

## 3. Kiểm thử hoặc tái lập phân tích offline khi cần

Chạy từ root bằng PowerShell. Không tải model, không chạy retrieval/GPU và không
mở reporting. Không chạy evaluator full-gold; script chỉ chấm gold trong diagnosis
export, calibration lấy aggregate đã có. Full-gold JSONL chỉ được hash byte để
kiểm tra tính toàn vẹn, không decode/parse. Bootstrap 10.000 draw, seed 42;
CI diagnosis 95% là exploratory, không phải kết quả holdout và không hiệu chỉnh
multiple comparisons. QASPER resample tất cả diagnosis paper rồi macro theo scored
query; SciFact resample claim. Group chỉ có query skipped vẫn tham gia resample;
draw có denominator 0 được loại và đếm.

```powershell
git status --short
git branch --show-current
$python = '.\.venv\Scripts\python.exe'
& $python -m unittest scripts.test_analyze_baseline_diagnosis -v
# v1 đã có: chỉ tái lập khi cần, dùng output-dir mới để không ghi đè.
& $python -m scripts.analyze_baseline_diagnosis --archive artifacts/artifacts_VERIFIED_COMPLETE_20261031T170355811993Z_1bcc0097.zip --sha256-file artifacts/artifacts_VERIFIED_COMPLETE_20261031T170355811993Z_1bcc0097.sha256.json --output-dir analysis/baseline_diagnosis_v2 --seed 42 --bootstrap-draws 10000 --review-size 60
```

Input bắt buộc:

- ZIP và sidecar ở trên: config, source snapshot/provenance, manifests, queries,
  documents, source BM25/dense top50, reranker retrieval/metrics development/diagnosis.
- `configs/dev_qasper_seed42.json`, `configs/dev_scifact_seed42.json`, giữ nguyên.
- Với cả `qasper` và `scifact`, mỗi thư mục sau phải có
  `<dataset>_retrieval.json`, `<dataset>_metrics_development.json`, `<dataset>_diagnosis.json`:
  `evaluation/reports/dev_bm25_v1/`, `dev_dense_gpu_v1/`,
  `dev_rrf_top10_v1/`, `dev_rrf_pool50_top10_v1/`.

Thiếu file hoặc hash/coverage/denominator không khớp thì dừng và bổ sung đúng
artifact gốc. Không reset/clean, không thay manifest hoặc sửa metadata để ép chạy.
Output độc lập với benchmark; không ghi đè report nguồn. File `output_hashes.json`
liệt kê SHA256 đầu ra, `provenance.json` ghi SHA256 từng input và nguồn code/model.

## 4. Chỉ một pilot sau human verification và cost checks

**Neighbor-one-swap**, chưa triển khai/chạy: từ top10 reranker, xét paragraph ±1
cùng numeric section trong original pool; chọn candidate chưa có theo RRF k60 từ
source ranks (tie evidence ID), thay đúng rank10 nếu candidate cost ≤ item bị thay.
Giữ nguyên text units/pool/model SHA, không thêm evidence thứ11, không tăng tổng
context tokens. Không thêm duplicate penalty/heading rule/2-hop hoặc sweep tham số.
Gold/review labels không được dùng trong inference. Luật đầy đủ, ví dụ hỗ trợ và
phản chứng nằm ở [MANUAL_REVIEW](MANUAL_REVIEW.md).

Thiếu tokenizer-specific costs và reader/token budget đã chốt: cần xác định cùng
cost rule cho baseline và selector trước khi chạy matched-token. Max-length512 của
cross-encoder không phải context budget reader. Full-union cross-encoder logits
không có, nhưng pilot dùng top10 và source ranks nên không cần tải/chạy lại model
để lấy logits đó. Không báo gold-assisted8/27 như cải thiện dự kiến.

Sau khi đủ điều kiện, freeze một protocol trước pilot và báo success regressions
cùng complete-hit/Recall/MRR; không chỉ xem phần thắng. Reporting/test vẫn giữ kín,
không phân tích lỗi calibration. Đợt hiện tại dừng ở review và đề xuất, không chạy
thí nghiệm selector hoặc phát triển controller/generation/UI.
