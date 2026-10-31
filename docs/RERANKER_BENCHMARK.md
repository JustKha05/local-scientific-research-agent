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

Model của phiên này được ghim ở SHA `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
Không resolve `main`, đổi SHA hoặc fallback CPU. Loader dùng pinned
`snapshot_download` rồi mở snapshot local, không thay classification head.
Source ghim theo commit `1d3ab1035545a867be1c649be17ad22565303e1f`;
notebook ghi commit và SHA256 từng file. Không tự commit/push.

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

## Chạy notebook A → G

**Lần đầu:** chọn kernel Colab GPU, chạy A để mount Drive, chuẩn bị input ZIP
đúng layout trên. Chỉ khi tạo thí nghiệm mới, đặt `CREATE_NEW=True`, `NEW_WORK`
và `NEW_DRIVE_RUN` bằng đường dẫn rõ ràng chưa tồn tại. B ghi config với SHA
cố định. Run All không tự tạo RUN_ID/workspace theo thời gian. C đồng bộ đúng
commit; D cài dependencies, kiểm tra CUDA và fixture tests (model giả).
Không chạy BM25/dense/RRF, không dùng `--include-reporting`.

**Tiếp tục phiên:** để `CREATE_NEW=False`. A đã chọn rõ config local của phiên
`20261031T164343Z_05af25c1`; không tự chuyển sang phiên khác nếu thiếu config.
B liệt kê config local và Drive;
nếu có nhiều đường dẫn, đặt `CONFIG_SELECTION` bằng đúng một đường dẫn rồi
chạy lại B. Phiên đã biết:
`/content/reranker_20261031T164343Z_05af25c1/reranker_run_config.json`.
Output cũ chỉ xác nhận QASPER reranking 1005 query, exit 0; chưa xác nhận
evaluation QASPER hoặc SciFact. B kiểm tra file thực tế, không tin output cũ.

Nếu mất runtime, chọn config Drive, đặt `RESTORE_ZIP` tới archive G và
`RESTORE_SHA256` từ sidecar `.sha256.json`. B kiểm tra hash ZIP, manifest,
config và từng file trước khi khôi phục. Archive cũ thiếu dữ liệu/source gốc
không đủ để resume: cần khôi phục đầy đủ workspace và đúng source đã tạo report.
Không xóa report hợp lệ để chạy lại.

Chạy tiếp C, D, E, F, G. C giữ nguyên byte source cũ trước khi đồng bộ.
E kiểm tra config, data/manifest, source reports, metadata và script hashes.
Hash script của report cũ phải khớp source đã giữ hoặc source ghim; không có
source tương ứng thì dừng. Sau kiểm tra đầu, seal SHA256 khóa report/config/input
để phát hiện thay đổi ở lần resume sau. Report hợp lệ đã có được giữ;
chỉ thiếu retrieval mới rerank. F chỉ gọi CLI evaluation khi thiếu metrics/diagnosis;
file đã có được kiểm tra nội dung bằng evaluator ghim, giới hạn diagnosis/calibration.
Output thiếu dựng trong staging; file hỏng hoặc xung đột gây lỗi, không ghi đè.

Mọi subprocess chạy với `cwd=WORK`, stream stdout/stderr và lưu log riêng trong
`resume_logs/`; traceback gốc tiến trình con hiện trong output và log.
Nếu E/F lỗi, sửa nguyên nhân rồi chạy lại; có thể chạy G riêng để lưu phần dở dang.
Lỗi được lưu riêng thành `FAILED_*.json` trên Drive. G dùng tên archive có timestamp
và suffix duy nhất, kiểm tra nội dung ZIP và lưu SHA256. Chỉ `VERIFIED_COMPLETE`
xác nhận cả hai dataset đã qua kiểm tra; `FAILED_OR_INCOMPLETE` là backup dở dang,
cell vẫn báo lỗi. ZIP chưa có sidecar hash hợp lệ không được dùng để restore.

**File gửi sau khi xong:** archive `artifacts_VERIFIED_COMPLETE_*.zip` mới nhất,
sidecar `.sha256.json` cùng tên và bảng trạng thái F. Archive gồm retrieval,
metrics development, diagnosis, config/SHA, source/provenance, manifest,
input data/source reports và logs để tái lập. Nếu lỗi, gửi backup
`FAILED_OR_INCOMPLETE` nếu tạo được, `FAILED_*.json` và log liên quan.
Diagnosis chứa gold; chỉ chia sẻ trong phạm vi nghiên cứu được phép.
