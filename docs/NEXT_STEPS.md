# Bước tiếp theo — chốt baseline và chuẩn bị benchmark dev

## 1. Đồng bộ đúng nhánh trên Windows
Tại thư mục project, kiểm tra trước:
```powershell
Set-Location D:\Projects\local-scientific-research-agent
git status
git fetch origin
git switch feat/dense-retrieval
git pull --ff-only origin feat/dense-retrieval
```
Nếu đang sửa file chưa commit, lưu công việc bằng commit hoặc stash do bạn lựa chọn trước khi đổi nhánh; không reset/clean. Nếu fast-forward thất bại, kiểm tra lịch sử thay vì ép ghi đè. main hiện chưa chứa pipeline retrieval.

## 2. Kiểm tra dữ liệu và tạo lại smoke nếu chưa có report
```powershell
.\.venv\Scripts\python.exe -m scripts.validate_processed_data
.\.venv\Scripts\python.exe -m scripts.run_bm25_smoke
.\.venv\Scripts\python.exe -m scripts.run_dense_smoke --dataset both
```
Hai lệnh retrieval ghi lại report trong bm25_smoke/dense_smoke. Nếu đã có report muốn giữ, sao lưu trước; nếu dữ liệu/revision/config không đổi và chỉ cần tái chấm thì bỏ qua hai lệnh retrieval.
Dense runner hiện chạy CPU và có cache; không cần Ollama cho bước này.

## 3. Tái chấm cùng 30 query train
```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_bm25_smoke --report-dir evaluation/reports/bm25_smoke
.\.venv\Scripts\python.exe -m scripts.evaluate_bm25_smoke --report-dir evaluation/reports/dense_smoke
.\.venv\Scripts\python.exe -m scripts.evaluate_scifact_rationales --report-dir evaluation/reports/bm25_smoke
.\.venv\Scripts\python.exe -m scripts.evaluate_scifact_rationales --report-dir evaluation/reports/dense_smoke
.\.venv\Scripts\python.exe -m scripts.inspect_bm25_errors
```
evaluate_bm25_smoke là evaluator dùng chung, không chạy BM25 lại. inspect_bm25_errors hiện chỉ tạo phân tích BM25, chưa hỗ trợ --report-dir.
File đầu ra: mỗi thư mục report có qasper_metrics.json, scifact_metrics.json, scifact_rationale_metrics.json. Mẫu review BM25: qasper_errors.md và scifact_errors.md.
Smoke không đủ 50–100 lỗi; review sớm vài trường hợp để debug, chưa dùng kết luận nghiên cứu.

## 4. Công việc lập trình tiếp theo (chưa được triển khai trong cập nhật tài liệu này)
1. Tạo runner benchmark tổng quát: --dataset, --split, --retriever, --model-config, --device, --top-k, --report-dir, --query-manifest.
2. Tổng quát hóa evaluator khỏi train/smoke, giữ nguyên metric cũ và validation để đối chiếu.
3. Cho runner dense chạy cpu/cuda và cache đúng dataset/split/model/tokenizer; chunk hoặc kiểm soát truncation có mapping về evidence gốc.
4. Tạo manifest diagnosis/calibration/reporting trước khi đọc dev errors; QASPER nhóm theo paper.
5. Chạy BM25/MiniLM toàn dev, giữ QASPER paper-scoped và SciFact corpus-scoped.
6. Xuất mẫu review 50–100 trường hợp: miss, partial, success; lấy mẫu có seed, ghi strata. Nếu oversample lỗi, không dùng tỷ lệ mẫu như tỷ lệ lỗi toàn dataset.

CLI của bước 4 là đặc tả sẽ xây, không phải lệnh có thể chạy ngay. Không thêm --split hoặc --device vào smoke script hiện tại.

## 5. Mẫu chẩn đoán
Mỗi dòng review cần: dataset, query_id, paper_id, retriever, evidence IDs, metric, category, observation, proposed_fix.
Category: lexical mismatch; semantic distractor; truncation; thiếu ngữ cảnh lân cận; evidence phân tán; annotation/representation; đủ evidence nhưng reader sai (chỉ gán sau khi có reader); unsupported modality; other/uncertain.
Ghi bằng chứng cho chẩn đoán, không suy luận lỗi reader khi chưa chạy generation.

## 6. Điều kiện quyết định hướng cải tiến
- Evidence thường không vào candidate pool: ưu tiên retriever/hybrid/hard negatives.
- Evidence có trong pool nhưng xếp thấp: ưu tiên reranker.
- Thiếu phần bổ trợ/đứt ngữ cảnh: kiểm tra context-aware selection.
- Reader sai dù có gold context: ưu tiên reader trước controller.
- Chỉ triển khai adaptive retrieval sau khi baseline mạnh và phân tích xác nhận lợi ích tiềm năng.

Thứ tự: xác nhận smoke → triển khai full dev → phân tích lỗi → hybrid/reranker → thử phương pháp → đánh giá cuối → demo.

