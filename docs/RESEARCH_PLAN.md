# Kế hoạch nghiên cứu — 29/10/2026

## 1. Mục tiêu và giới hạn tuyên bố
Xây dựng Scientific QA có kiểm tra bằng chứng, quyết định trả lời / tìm bổ sung / từ chối dưới ngân sách tính toán giới hạn. Mục tiêu là cải thiện có thể tái lập so với baseline mạnh, đủ hữu ích cho người đọc bài báo. Chưa tuyên bố phương pháp mới, SOTA hay đã giảm hallucination; cần rà soát related work và thực nghiệm.

QASPER là tác vụ chính: QA trong một bài báo đã biết. SciFact là tác vụ mở rộng: truy xuất và xác minh claim trên corpus. Hai tác vụ có evaluator riêng; không gộp điểm hoặc coi SciFact thiếu rationale là QASPER unanswerable.

## 2. Hiện trạng kiểm tra local — 01/11/2026

Nguồn theo dõi chính: [PROGRESS.md](PROGRESS.md); nhánh hiện tại
`feat/dev-retrieval-benchmark`, HEAD kiểm tra `83a5ee39cdb839400f040dd503fcbaff48c11576`.
Không suy diễn trạng thái các nhánh remote từ kiểm tra local này.

- Đã triển khai: data preparation/validation, full-dev runner cpu/cuda, manifest
  diagnosis/calibration/reporting, evaluator, BM25, dense MiniLM, RRF và cross-encoder
  ghim revision; notebook resume/provenance/backup và script phân tích offline.
- Artifact đã có: full-dev predictions QASPER 1005 query, SciFact 300 query;
  BM25/dense top10 và top50, hai biến thể RRF top10, reranker top10.
  Diagnosis/calibration metrics và diagnosis exports đã có; reporting giữ kín.
- Reranker chạy Colab Tesla T4, source `1d3ab1035545a867be1c649be17ad22565303e1f`,
  model SHA `233902d25c440f23af6f7d6e94d2946bac0bee0a`; archive local có SHA256
  được xác minh lại. Nhãn `VERIFIED_COMPLETE` chỉ xác nhận hoàn tất/toàn vẹn.
- Phân tích offline kiểm tra hashes, manifest/query coverage và denominator;
  chấm riêng diagnosis, bootstrap paired theo paper/claim. Calibration chỉ lấy
  aggregate sẵn có. Xem số liệu và giới hạn trong PROGRESS/báo cáo offline.
- Semantics giữ nguyên: QASPER best recall theo annotation hợp lệ, normalized
  duplicate text tương đương nhưng vẫn chiếm rank; SciFact recall dùng union,
  complete hit chỉ cần một rationale. Không phải official QA/verification scores.
- Chưa làm: manual review hoàn chỉnh, embedding mạnh hơn, selector evidence mới,
  reader/generation, oracle reader, controller, final reporting/test, demo.

## 3. Câu hỏi nghiên cứu
RQ1: Khi cùng retriever/reader và ngân sách token, chọn bằng chứng có xét ngữ cảnh và trùng lặp có cải thiện độ đầy đủ bằng chứng và Answer F1?
RQ2: Controller tìm bổ sung / trả lời / từ chối có cải thiện quan hệ accuracy–coverage so với luôn trả lời và luôn truy xuất thêm?
RQ3 (mở rộng): Cách chọn bằng chứng có giúp SciFact rationale và verification không?

Chỉ triển khai controller sau khi phân tích lỗi cho thấy thiếu bằng chứng là điểm nghẽn đáng kể. Nếu gold-context reader vẫn yếu, ưu tiên reader/prompt trước.

## 4. Các giai đoạn và điều kiện hoàn thành
| Giai đoạn | Việc thực hiện | Điều kiện chuyển bước |
|---|---|---|
| A. Chốt baseline hiện có | Kiểm tra artifact full-dev đã có, revision/hash/config và so sánh cùng query | Hai retriever có report hợp lệ; nêu rõ scored/skipped |
| B. Benchmark dev tổng quát | Runner/evaluator hỗ trợ split, dataset, model, device, top-k, output; không phá smoke | Full dev chạy được; phạm vi paper/corpus rõ; kiểm tra ID, rank và cache |
| C. Chẩn đoán | Review 50–100 trường hợp từ dev diagnosis, gồm miss/partial/success và nhiều loại câu hỏi | Có bảng tỷ lệ lỗi, ví dụ và quyết định điểm nghẽn |
| D. Baseline mạnh | BM25, MiniLM, một embedding mạnh hơn, hybrid RRF, hybrid + cross-encoder | So sánh cùng split, candidate pool, context budget và reader |
| E. Phương pháp cải tiến | Chọn bằng chứng theo ngữ cảnh/giảm trùng; controller có giới hạn lượt và token | Có lợi ích đo được so với baseline mạnh; có ablation |
| F. Đánh giá cuối | QASPER test được giữ kín; SciFact dev reporting holdout hoặc chấm test chính thức nếu có | Đóng băng cấu hình; CI; phân tích lỗi và chi phí |
| G. Sản phẩm/release | API/demo mở nguồn bằng chứng, log, hướng dẫn chạy GPU/local, báo cáo | Một lệnh tái lập benchmark; ví dụ thành công và thất bại |

B và phần BM25/MiniLM/RRF/reranker của D đã có code và artifact full-dev.
C hiện có phân tích định lượng offline; manual review và quyết định ablation còn lại.
Embedding mạnh hơn, reader và so sánh ngân sách token ở D chưa hoàn tất.
Các giai đoạn E–G vẫn là kế hoạch, chưa phải kết quả đã thực hiện.

## 5. Giao thức tránh leakage
- Train: fine-tune, hard-negative mining và xây tập huấn luyện controller.
- Trước khi xem lỗi dev, cố định manifest chia dev thành diagnosis / calibration / reporting. QASPER chia theo PAPER để câu hỏi cùng paper không rơi qua các phần.
- Diagnosis dùng phân tích lỗi; calibration dùng chọn threshold/hyperparameter. Reporting không dùng chỉnh mô hình.
- QASPER test chỉ đánh giá cuối. Nếu cần dùng lại nhiều lần phải công bố đó không còn là test giữ kín.
- SciFact test gốc không có gold công khai. Đánh giá cục bộ trên reporting dev và công bố đúng số query; không gọi đó là kết quả test toàn bộ.
- Benchmark BEIR SciFact là document retrieval với split/qrels riêng. Không đánh đồng tên split của BEIR với split gốc.
- Gold chỉ dùng train/evaluation/oracle diagnosis; không đưa gold vào controller inference.
- Không coi mọi evidence chưa annotate là negative đúng; kiểm tra false negatives khi mining.
- Giữ unanswerable QASPER và claim không có evidence SciFact cho đánh giá đầu ra; chỉ loại khỏi metric retrieval yêu cầu positive, kèm denominator và lý do.

## 6. Thực nghiệm chính
Baseline: BM25; dense MiniLM; thêm một embedding sau kiểm tra tài nguyên; hybrid RRF; hybrid + reranker.
Reader: một model chính, giữ nguyên prompt/decoding; model thứ hai chỉ kiểm tra độ ổn định nếu đủ nguồn lực.
So sánh controller với: luôn trả lời; từ chối theo threshold tĩnh; luôn truy xuất bổ sung; adaptive.
Ablation: bỏ mở rộng ngữ cảnh; bỏ giảm trùng; bỏ controller; bỏ lượt tìm bổ sung.
Oracle gold context chỉ dùng chẩn đoán giới hạn reader, tách riêng khỏi kết quả thực tế.

Ngân sách dự kiến 1k/2k/4k token context theo tokenizer reader, tính cả citation marker và ngữ cảnh lân cận. Chốt sau pilot và trước reporting. Với adaptive, cộng token/tính toán của mọi lượt và verifier; giới hạn tối đa 2 lượt retrieval bổ sung. Báo cáo cả chất lượng theo tổng chi phí và chất lượng theo final-context budget.

## 7. Metrics
- Retrieval nội bộ: Recall@1/5/10, MRR@10; giữ nguyên mô tả annotation/duplicate policy.
- Evidence: complete evidence/rationale hit; precision/recall/F1 theo đúng đơn vị và evaluator.
- QASPER QA: official Answer F1 và Evidence F1 khi xuất được đúng schema; tách text-only và all-question. Figure/table chưa hỗ trợ phải nêu rõ.
- SciFact: official abstract/sentence label/evidence metrics khi có adapter; diagnostic sentence recall vẫn giữ riêng.
- Selective QA: accuracy/F1 theo coverage; risk–coverage curve; phân biệt abstention của hệ thống với nhãn unanswerable.
- Citation: kiểm tra câu trả lời có được bằng chứng hỗ trợ, dùng rubric/manual sample; evidence overlap không tự chứng minh factual support.
- Cost: indexing/encoding riêng với query; p50/p95 latency, RAM/VRAM, token và số lượt; ghi hardware/cache/warmup.
- Paired bootstrap CI cho chênh lệch; QASPER resample theo paper. Fine-tune ít nhất 3 seed nếu khả thi. Không đặt mức tăng tùy ý rồi coi đó là kết quả.

## 8. So sánh công trình có sẵn
- QASPER LED: https://github.com/allenai/qasper-led-baseline — official QA/evidence evaluator.
- VeriSci: https://github.com/allenai/scifact — pipeline claim verification và evaluator.
- MultiVerS: https://github.com/dwadden/multivers — checkpoint/baseline verification; ghi rõ pretraining ngoài dataset.
- BEIR: https://github.com/beir-cellar/beir — document retrieval; không so với top-k câu.
- LongBench: https://github.com/THUDM/LongBench — track riêng nếu chạy đúng phiên bản QASPER của benchmark.

Chọn VeriSci hoặc MultiVerS làm đối chứng SciFact tùy khả năng chạy; không bắt buộc tái huấn luyện mọi công trình. Điểm paper chỉ nằm bảng tham khảo khi protocol không khớp. Rà soát literature về adaptive retrieval, evidence sufficiency và selective QA trước khi tuyên bố novelty.

## 9. GPU và tái lập
Máy Windows: dữ liệu, debug, smoke, demo. Colab/Kaggle: encode/full benchmark/fine-tune khi runner hỗ trợ GPU. Không phụ thuộc tên GPU hay quota cố định.
Lưu manifest dataset/query IDs, commit, model revision, tokenizer/truncation, config, seed, package versions, predictions, metrics, hardware và checkpoint resume.
Tách requirements CPU/GPU khi triển khai; kiểm tra torch CUDA thực tế. Không đưa token/credential vào notebook hoặc git.
Không chạy lại smoke CPU để giả lập benchmark GPU. So latency chỉ khi cùng hardware/protocol.

## 10. Bước ngay bây giờ

Xem [PROGRESS.md](PROGRESS.md) và [NEXT_STEPS.md](NEXT_STEPS.md). Review baseline
và ví dụ diagnosis trước khi thử một evidence-selection ablation. Không chạy lại
retrieval/model/GPU, không mở reporting trong đợt phân tích này. Chưa fine-tune,
thêm multi-agent hoặc làm UI lớn trước khi có chẩn đoán.

Đóng góp mục tiêu: phương pháp và bằng chứng thực nghiệm về cách dùng context/tính toán hiệu quả để trả lời có căn cứ. Nếu không thắng baseline, công bố giới hạn và kết quả âm trung thực; không đổi test thành tập tuning.

