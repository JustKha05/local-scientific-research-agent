# Cross-encoder reranker benchmark

Lượt đầu dùng `cross-encoder/ms-marco-MiniLM-L6-v2`, batch 16, max length 512,
top-k 10, float32, raw relevance logit. Đây là baseline pretrained MS MARCO,
không fine-tune QASPER/SciFact. Xem [model](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
và [CrossEncoder API](https://sbert.net/docs/package_reference/cross_encoder/model.html).
Repo dùng `sentence-transformers==6.0.1` trong requirements; không thay baseline cũ.

Runner chỉ đọc prediction report. Nó tái sử dụng `validate_sources` của RRF để
kiểm tra dataset/split/source hashes/manifest hash/query IDs/text/scope/candidate_count
và toàn bộ payload evidence (ngoài rank/score), kể cả nội dung xung đột giữa query.
Không đọc gold, manifest assignments hay partition để xếp hạng.

Mỗi nguồn phải cung cấp ít nhất `min(50, candidate_count)` hit mỗi query. Runner
lấy đúng top 50 mỗi nguồn, union theo evidence ID, tối đa 100, chấm tất cả bằng
cặp query + text gốc, rồi sort score giảm dần và evidence ID tăng dần khi hòa.
Không lấy candidate từ output RRF top 10. Source sâu hơn 50 được cắt theo rank;
source top 10 bị từ chối nếu scope còn đủ candidate. Dùng hai thư mục có sẵn:
`evaluation/reports/dev_bm25_top50_v1` và `evaluation/reports/dev_dense_top50_v1`.
Nếu thiếu, tạo report top 50 mới bằng retrieval runner theo giao thức hiện tại;
không sửa report top 10 và không đổi manifest.

Revision phải là commit SHA 40 ký tự. Resolve **một lần trước lượt chạy**, lưu lại,
rồi dùng cùng SHA cho cả hai dataset. Không resolve lại `main` giữa hai lượt.
Ví dụ PowerShell từ root repo (truy vấn metadata không tải weights):

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m unittest scripts.test_rerank_retrieval scripts.test_fuse_retrieval_rrf -v

# Chỉ chạy resolve khi bắt đầu một cấu hình mới; file tồn tại sẽ báo lỗi.
@'
from huggingface_hub import model_info
from scripts.benchmark_common import write_new
m = "cross-encoder/ms-marco-MiniLM-L6-v2"
write_new("evaluation/reranker_model_v1.json", {
    "model_id": m, "revision": model_info(m, revision="main").sha,
    "device": "cuda", "batch_size": 16, "max_length": 512, "top_k": 10})
'@ | & $python -
if ($LASTEXITCODE -ne 0) { throw "Resolve failed; reuse the existing config or choose a new file" }
$cfg = Get-Content evaluation/reranker_model_v1.json -Raw | ConvertFrom-Json
foreach ($dataset in @('qasper', 'scifact')) {
    & $python -m scripts.rerank_retrieval --dataset $dataset --bm25-dir evaluation/reports/dev_bm25_top50_v1 --dense-dir evaluation/reports/dev_dense_top50_v1 --report-dir evaluation/reports/dev_cross_encoder_v1 --model-id $cfg.model_id --revision $cfg.revision --device $cfg.device --batch-size $cfg.batch_size --max-length $cfg.max_length --top-k $cfg.top_k
    if ($LASTEXITCODE -ne 0) { throw "Reranking failed" }
    & $python -m scripts.evaluate_retrieval_benchmark --dataset $dataset --split dev --query-manifest "configs/dev_${dataset}_seed42.json" --report-dir evaluation/reports/dev_cross_encoder_v1
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed" }
}
```

CPU: lưu `device=cpu` khi tạo cấu hình mới. CUDA được kiểm tra trước model load;
không fallback CPU ngầm. CLI không thay head thành một output: model nhiều output
bị từ chối. Shape score phải `(N,)` hoặc `(N,1)` và mọi score hữu hạn.
Evaluator cần top-k >= 10. Không dùng `--include-reporting`.

Payload evidence được giữ nguyên, chỉ thay rank/score. Report lưu config model và
tokenizer cùng SHA, tokenizer class, float32, versions, hardware, hash hai file
nguồn và script validation/runner, độ sâu nguồn và số candidate thực sự được chấm.
`candidate_count` vẫn là kích thước scope gốc để tương thích evaluator;
`rerank_candidate_count` là kích thước union. Tokenizer cắt pair bằng longest-first
từ bên phải, gồm special tokens; không chunk, không thay text trong payload.
Thống kê pair token trước truncation gồm số vượt giới hạn, độ dài lớn nhất và
tổng token vượt giới hạn, theo query và toàn report.

`search_seconds` và `rerank_seconds` theo query chỉ đo tạo pair, tokenizer,
inference, sort/top-k; đồng bộ CUDA ở hai đầu, không warmup. Tổng `rerank_seconds`
không gồm retrieval nguồn, union/validation, audit truncation, model load hay I/O.
Không gọi đây là latency toàn pipeline, không cộng tùy tiện vào latency baseline
khác hardware. Model load được ghi riêng. Peak VRAM là torch allocated memory.
File output tồn tại sẽ gây lỗi; dùng report-dir mới cho lượt mới.

## ZIP đầu vào và VSCode + Colab

Notebook [reranker_benchmark_gpu.ipynb](../notebooks/reranker_benchmark_gpu.ipynb)
chỉ gọi CLI của repo. ZIP phải có các đường dẫn dưới đây ngay ở root, không bọc
thêm thư mục project. Chỉ đóng gói nội dung cần thiết, không `.venv`, `.git`,
credentials, raw data hay metrics/diagnosis/reporting output cũ.

```text
scripts/*.py
app/*.py
requirements.txt
configs/dev_qasper_seed42.json
configs/dev_scifact_seed42.json
data/processed/qasper/dev/{queries,documents,gold}.jsonl
data/processed/scifact/documents.jsonl
data/processed/scifact/dev/{queries,gold}.jsonl
evaluation/reports/dev_bm25_top50_v1/{qasper,scifact}_retrieval.json
evaluation/reports/dev_dense_top50_v1/{qasper,scifact}_retrieval.json
```

Gold chỉ được evaluator đọc sau reranking; runner không truy cập file đó.
Tạo ZIP input mới bằng PowerShell (không thay dữ liệu/report nguồn):

```powershell
$stage = Join-Path $env:TEMP ("reranker_input_" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $stage | Out-Null
foreach ($dir in @('scripts','app')) {
    New-Item -ItemType Directory -Path "$stage/$dir" | Out-Null
    Copy-Item "$dir/*.py" "$stage/$dir/"
}
Copy-Item requirements.txt $stage
$paths = @('configs/dev_qasper_seed42.json','configs/dev_scifact_seed42.json',
    'data/processed/qasper/dev/queries.jsonl','data/processed/qasper/dev/documents.jsonl',
    'data/processed/qasper/dev/gold.jsonl','data/processed/scifact/documents.jsonl',
    'data/processed/scifact/dev/queries.jsonl','data/processed/scifact/dev/gold.jsonl')
foreach ($method in @('bm25','dense')) {
    foreach ($dataset in @('qasper','scifact')) {
        $paths += "evaluation/reports/dev_${method}_top50_v1/${dataset}_retrieval.json"
    }
}
foreach ($path in $paths) {
    $target = Join-Path $stage $path
    New-Item -ItemType Directory -Path (Split-Path $target) -Force | Out-Null
    Copy-Item -LiteralPath $path -Destination $target
}
$zip = Join-Path (Get-Location) 'reranker_benchmark_input_v1.zip'
if (Test-Path -LiteralPath $zip) { throw "ZIP exists; choose a new name" }
Compress-Archive -Path "$stage/*" -DestinationPath $zip
```

Chép ZIP lên `MyDrive/retrieval_benchmark/` bằng Drive web/desktop. Trong VSCode
chọn kernel Colab có GPU. Mount Drive qua `google.colab.drive`; nếu runtime yêu
cầu xác thực trong giao diện Colab, mount một lần ở Colab trên cùng runtime rồi
quay lại VSCode. Sửa `INPUT_ZIP`, `DRIVE_OUTPUT` trong notebook nếu cần.
Không dùng widget upload/download.

Notebook giải nén vào thư mục mới, kiểm tra CUDA, cài requirements bằng Python
của kernel, resolve SHA bằng `huggingface_hub.model_info`, lưu config lên Drive
**trước** khi chạy. Nó gọi `scripts.rerank_retrieval` rồi evaluator mặc định cho
QASPER/SciFact. Không chạy lại retrieval và không mở reporting. Output ZIP trên
Drive chứa report mới, config/SHA, manifest nguyên bản và source snapshot để
tái lập; không gộp report cũ. Sao lưu sau từng dataset, kể cả khi evaluator lỗi.

Colab terminal tương đương sau giải nén/cài dependencies và resolve config:

```bash
python -m scripts.rerank_retrieval --dataset qasper --bm25-dir evaluation/reports/dev_bm25_top50_v1 --dense-dir evaluation/reports/dev_dense_top50_v1 --report-dir evaluation/reports/dev_cross_encoder_v1 --model-id cross-encoder/ms-marco-MiniLM-L6-v2 --revision "$MODEL_SHA" --device cuda --batch-size 16 --max-length 512 --top-k 10
python -m scripts.evaluate_retrieval_benchmark --dataset qasper --query-manifest configs/dev_qasper_seed42.json --report-dir evaluation/reports/dev_cross_encoder_v1
# Lặp lại hai lệnh với --dataset scifact và manifest dev_scifact_seed42.json.
```

`MODEL_SHA` phải lấy từ config đã lưu; không đặt bằng `main`. Test fixture chỉ dùng
model giả và không xác nhận chất lượng benchmark hay hoạt động CUDA thực tế.
