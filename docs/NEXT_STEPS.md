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

## 2. Review thủ công diagnosis

Dùng [MANUAL_REVIEW.md](../analysis/baseline_diagnosis_v1/MANUAL_REVIEW.md)
(60 query/dataset, phân tầng theo bucket, seed 42) và danh sách đầy đủ
`diagnosis_queries.csv`. Mỗi ví dụ cần category do người review gán, observation,
bằng chứng cụ thể và proposed fix. Không tự suy ra lexical/semantic/truncation
hoặc lỗi reader từ metric. Mẫu phân tầng không ước lượng tỷ lệ lỗi quần thể.
Các tỷ lệ bucket trong báo cáo dùng toàn bộ scored diagnosis.

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

## 4. Quyết định thí nghiệm tiếp theo sau manual review

Đọc quyết định cập nhật trong [PROGRESS.md](PROGRESS.md). Chỉ thử một ablation
chọn evidence sau khi kiểm tra ví dụ. Giữ pool, model SHA và ngân sách top10/token
cố định để tách ảnh hưởng selector khỏi retriever. Không dùng gold trong inference.
Calibration chỉ dùng aggregate hiện có trong bước này; trước một lần tuning tương
lai phải chốt protocol. Reporting và test tiếp tục giữ kín.
Chưa phát triển controller, generation hoặc UI lớn trước khi có chẩn đoán rõ.
