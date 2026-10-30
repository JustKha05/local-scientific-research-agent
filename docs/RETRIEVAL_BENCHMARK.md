# Benchmark retrieval toàn dev

Các lệnh dưới đây dành cho runner mới; không thêm tham số vào smoke script.
Smoke và kết quả cũ được giữ nguyên. Benchmark chưa được chạy trên dữ liệu thật trong lần triển khai này.

Đã lưu hai manifest trong `configs/dev_qasper_seed42.json` và `configs/dev_scifact_seed42.json`
từ query/paper ID processed hiện tại, không đọc gold. Giữ nguyên và dùng lại:

| Dataset | Diagnosis query/group | Calibration query/group | Reporting query/group |
|---|---:|---:|---:|
| QASPER | 416 / 113 | 206 / 56 | 383 / 112 |
| SciFact | 120 / 120 | 60 / 60 | 120 / 120 |

## Giao thức

- QASPER: candidate là mọi paragraph trong paper của query, BM25 có index riêng từng paper.
- SciFact: candidate là mọi câu trong corpus processed, không lọc theo gold document.
- Runner chỉ đọc queries/documents, không mở gold. MiniLM cố định revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, max_seq_length=256 như report dense smoke hiện tại.
  Không chunk, giữ đầu evidence gốc; đếm truncation cho cả evidence và query.
- Recall@1/5/10 và MRR@10 giữ semantics smoke: QASPER lấy best recall giữa annotation
  text-only hợp lệ, MRR dùng union; text chuẩn hóa trùng trong paper được xem tương đương.
  Mỗi hit vẫn chiếm một rank. SciFact recall theo union câu; complete rationale chỉ cần một set đầy đủ.
- Complete evidence hit của QASPER cũng chỉ cần một annotation hợp lệ đầy đủ.
  `complete_alternative` nghĩa là đủ một set nhưng chưa lấy hết union của mọi set;
  không coi đây là retrieval thất bại chỉ vì union recall < 1.
- Macro average trên scored queries; skip QASPER `no_eligible_text_annotation` và SciFact
  `no_annotated_evidence`, ghi số lượng riêng từng partition. Đây là diagnostic retrieval, không phải official QA/verification scoring.
- Manifest dev chia theo group với seed 42, tỷ lệ 40/20/40 theo số group, dùng largest remainder
  (tie ưu tiên diagnosis rồi calibration rồi reporting). QASPER group=paper; SciFact group=claim/query ID.
  Tỷ lệ query QASPER có thể khác tỷ lệ group. Counts thực tế nằm trong manifest; kiểm tra không giao group,
  coverage query và hash mỗi lần dùng. Manifest chỉ được tạo bằng lệnh riêng; file tồn tại gây lỗi.
- Report/cache không ghi đè. Nếu cần chạy lại, chọn report-dir mới. Cache lỗi hoặc thiếu một file sẽ dừng;
  chuyển cache hỏng sang nơi khác một cách tường minh trước khi tạo lại. Không sửa metadata để ép dùng cache.
  Cache key chứa config, versions, hash documents và evidence ID theo đúng thứ tự; kiểm tra hash vector,
  shape, dtype, finite và normalization. Cache riêng với smoke.

## PowerShell

Chạy từ root repo. Cài dependencies theo `requirements.txt` trong môi trường của bạn nếu chưa có.
CPU BM25 không cần GPU. Dense lần đầu có thể tải model khi **bạn** chạy; triển khai/test fixture không tải model.

```powershell
Set-Location D:\Projects\local-scientific-research-agent
$python = '.\.venv\Scripts\python.exe'
& $python -m unittest scripts.test_retrieval_benchmark -v

# Repo đã có hai manifest: dùng lại, không chạy lại bước tạo.
# Chỉ nếu bắt đầu một protocol mới CHƯA có manifest, trước khi xem lỗi dev:
# & $python -m scripts.create_dev_manifest --dataset qasper --output configs/dev_qasper_seed42.json
# & $python -m scripts.create_dev_manifest --dataset scifact --output configs/dev_scifact_seed42.json

foreach ($dataset in @('qasper', 'scifact')) {
    $manifest = "configs/dev_${dataset}_seed42.json"
    & $python -m scripts.run_retrieval_benchmark --dataset $dataset --split dev --retriever bm25 --device cpu --top-k 10 --query-manifest $manifest --report-dir evaluation/reports/dev_bm25_v1
    & $python -m scripts.evaluate_retrieval_benchmark --dataset $dataset --query-manifest $manifest --report-dir evaluation/reports/dev_bm25_v1
}
```

Dense CPU (có thể chậm) hoặc CUDA nếu torch đã cài đúng bản CUDA:

```powershell
& $python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
foreach ($dataset in @('qasper', 'scifact')) {
    $manifest = "configs/dev_${dataset}_seed42.json"
    & $python -m scripts.run_retrieval_benchmark --dataset $dataset --split dev --retriever dense --device cuda --top-k 10 --batch-size 16 --query-manifest $manifest --cache-dir data/embeddings/benchmark --report-dir evaluation/reports/dev_dense_v1
    & $python -m scripts.evaluate_retrieval_benchmark --dataset $dataset --query-manifest $manifest --report-dir evaluation/reports/dev_dense_v1
    & $python -m scripts.compare_retrieval_benchmarks --dataset $dataset --bm25-dir evaluation/reports/dev_bm25_v1 --dense-dir evaluation/reports/dev_dense_v1 --output-dir evaluation/reports/comparison_v1 --sample-size 100
}
```

Thay `--device cuda` bằng `--device cpu` một cách tường minh nếu muốn chạy dense CPU.
Không thay `device` trong cấu hình smoke. Runner GPU encode bằng GPU; vector được lưu float32 trên CPU,
dot product và stable top-k chạy NumPy trên CPU như baseline. Không gọi đây là end-to-end GPU retrieval.
BM25 từ chối device=cuda. Runner hỗ trợ train/dev/test; evaluator cần gold nên SciFact test gốc không chấm local được.
Evaluator cần top-k >= 10 để không báo sai Recall@10 khi chỉ lấy ít hit.

## Reporting holdout và phân tích lỗi

Runner chạy tất cả query nhưng evaluator mặc định chỉ chấm diagnosis/calibration.
Chỉ diagnosis được xuất query, hits và gold trong `*_diagnosis.json` và mẫu review Markdown.
Không mở retrieval JSON của reporting để chọn phương pháp; file prediction toàn dev là artifact cho chấm sau.
Calibration chỉ xuất số tổng hợp; reporting không có metric mặc định.

Sau khi **chốt phương pháp/config**, lưu quyết định đó cùng artifact rồi chấm:

```powershell
foreach ($dataset in @('qasper', 'scifact')) {
    foreach ($method in @('bm25', 'dense')) {
        & $python -m scripts.evaluate_retrieval_benchmark --dataset $dataset --query-manifest "configs/dev_${dataset}_seed42.json" --report-dir "evaluation/reports/dev_${method}_v1" --include-reporting
    }
    & $python -m scripts.compare_retrieval_benchmarks --dataset $dataset --bm25-dir evaluation/reports/dev_bm25_v1 --dense-dir evaluation/reports/dev_dense_v1 --output-dir evaluation/reports/comparison_v1 --include-reporting
}
```

`*_metrics_final.json` bổ sung metric reporting, vẫn không xuất nội dung lỗi reporting.
Bảng CSV chứa từng metric, hai baseline và delta; kiểm tra cùng manifest, input/gold hash và denominator.
Review lấy mẫu round-robin theo retriever/status, seed 42; có success/alternative/partial/miss/skipped nếu tồn tại.
Mẫu oversample không dùng để ước lượng tỷ lệ lỗi quần thể. Ghi category, observation, supporting evidence,
proposed_fix; chưa chạy reader thì không kết luận lỗi reader. Muốn tỷ lệ status toàn diagnosis dùng summary.

## Colab / Kaggle

Mở `notebooks/retrieval_benchmark_gpu.ipynb`. Bật GPU trong Runtime/Accelerator trước khi chạy.
Upload source repo, `data/processed` và **manifest đã tạo**, không upload `.venv`, `.git`, credential,
raw data hoặc toàn bộ thư mục máy. Không tái chia dev trên notebook nếu đã có manifest.
Colab dùng Files/Drive để đưa file lên; Kaggle dùng private Dataset/Input rồi copy vào `/kaggle/working`.
Chỉnh `REPO`, `PROCESSED`, `MANIFEST_DIR`, `CACHE` và `REPORT` ở cell cấu hình.
Nếu hết VRAM giảm batch-size, dùng report-dir mới và giữ manifest. Notebook dừng rõ ràng nếu thiếu GPU.
Không có token hay credential trong notebook. Cài torch CUDA theo môi trường notebook có sẵn;
notebook không tự thay torch nếu CUDA không hoạt động.

Notebook gọi lại module repo để retrieval/evaluation, lưu cache ra đường dẫn cấu hình và đóng ZIP gồm
reports, manifest và cache. ZIP có prediction/gold diagnosis; lưu riêng, không công khai reporting trước khi chốt.
Colab cho tải trực tiếp; Kaggle tải từ Output. Lưu cache vào Drive hoặc tải ZIP trước khi session hết hạn.

## Provenance và kiểm tra

Report ghi input byte hashes, manifest hash, revision, tokenizer length/truncation, cấu hình, package versions,
OS/CPU/GPU, commit/dirty, hash runner, index/cache time, tổng runtime, p50/p95 query latency.
Không warmup; query latency gồm encode, scoring CPU, chọn top-k; index/model load/cache tách riêng.
GPU peak allocated memory là chỉ số torch, không phải tổng VRAM của hệ thống. Không so latency khác hardware
hoặc cache condition như cùng một protocol. Lưu source snapshot cùng ZIP nếu git dirty.

Fixture test không instantiate SentenceTransformer: scope paper/corpus, runner không cần gold,
manifest cố định/không overwrite/không giao group, alternative rationale, cache metadata/ID,
validation cả query skipped, reporting gate và parity trực tiếp với hai evaluator smoke trên dữ liệu giả.
Chưa xác nhận throughput/chất lượng toàn dev hoặc CUDA runtime thực tế bằng các fixture này.
