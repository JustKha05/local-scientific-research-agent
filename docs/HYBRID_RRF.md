# Hybrid RRF từ prediction report

Chạy từ root repo bằng PowerShell. Không cần model, GPU hay gold để fusion.
Script chỉ đọc hai file `<dataset>_retrieval.json`; giữ nguyên baseline và manifest.

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m unittest scripts.test_fuse_retrieval_rrf -v

foreach ($dataset in @('qasper', 'scifact')) {
    & $python -m scripts.fuse_retrieval_rrf --dataset $dataset --bm25-dir evaluation/reports/dev_bm25_v1 --dense-dir evaluation/reports/dev_dense_gpu_v1 --report-dir evaluation/reports/dev_rrf_top10_v1 --rrf-k 60 --top-k 10
}
```

Mỗi query hợp evidence theo `evidence_id`. Điểm bằng tổng `1 / (60 + rank)`
chỉ từ nguồn thực sự trả về evidence đó. Hai nguồn có thể có độ sâu khác nhau;
fusion chỉ dùng hit đã lưu, không truy hồi thêm. Trùng ID khác payload evidence
(mọi trường ngoài rank/score) gây lỗi, kể cả giữa các query. ID trùng trong cùng
một danh sách nguồn cũng gây lỗi. Phá hòa bằng ID tăng dần, lấy top-k và đánh rank lại.

Report lưu SHA-256 của chính bytes hai file nguồn, top-k cấu hình và số hit thực tế
theo từng query của mỗi nguồn. File output tồn tại sẽ gây lỗi; chọn thư mục mới
nếu cần chạy lại. Không sửa hoặc tái tạo manifest.

`search_seconds` chỉ đo hợp candidate, tính điểm, sort và lấy top-k; không gồm
retrieval nguồn, đọc/ghi report hoặc validation. `fusion_seconds` là tổng thời gian
fusion các query. Các số này không phải latency toàn pipeline và không dùng để
so sánh trực tiếp với latency retrieval BM25/dense.

Đánh giá tùy chọn bằng evaluator hiện tại (đọc gold ở bước đánh giá riêng).
Chỉ diagnosis/calibration được chấm; không bật reporting:

```powershell
foreach ($dataset in @('qasper', 'scifact')) {
    & $python -m scripts.evaluate_retrieval_benchmark --dataset $dataset --split dev --query-manifest "configs/dev_${dataset}_seed42.json" --report-dir evaluation/reports/dev_rrf_top10_v1
}
```

Evaluator cần `--top-k >= 10`; CLI fusion cho phép top-k nhỏ hơn để dùng riêng.
Các lệnh fusion/đánh giá trên là hướng dẫn, không nằm trong kiểm tra fixture.
