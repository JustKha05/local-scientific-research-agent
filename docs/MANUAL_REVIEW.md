# QASPER diagnosis review — 01/11/2026

## Kết luận và bước tiếp theo

Đã tạo **40 lượt chọn nhóm, 37 query duy nhất thuộc 31 paper**: 20 selection_loss,
10 regression theo complete-evidence@10, 10 success đối chứng. Ba query cùng nằm
trong selection_loss và regression; không đếm hai lần khi báo thống kê unique.
Các packet liên kết bên dưới hiển thị question, mọi annotation gold hợp lệ,
top10 reranker/RRF pool50 (rank, score, text preview và liên kết full text),
và **toàn bộ candidate union top50** kèm full text, source rank/score, vị trí đoạn,
đánh dấu gold-equivalent và đoạn lân cận. Không lọc pool chỉ theo keyword.

**Nhãn do người xác minh: 0/37, tất cả PENDING.** Các số đếm là quan sát tự động;
nhận định nội dung dưới đây là trợ lý đọc văn bản, chưa phải nhãn người kiểm tra.
Không có causal label “semantic distractor”, “truncation” hoặc “reader error” nào
được xác nhận. Không chạy model/retrieval/GPU hoặc mở reporting/calibration errors.

Tín hiệu đủ để đề xuất **một pilot thay tối đa một evidence bằng đoạn lân cận
trong cùng pool, dưới cùng ngân sách**, chưa đủ chứng minh hiệu quả. Không đề xuất
thêm penalty trùng lặp trong pilot này: mẫu không cho thấy duplicate top10 theo
hai phép kiểm tra định trước. Không triển khai hoặc chạy selector trong đợt review.

## Cách chọn, định nghĩa và giới hạn

Seed=42; với mỗi nhóm dùng `random.Random(42).sample` trên population đã sort theo
query ID, độc lập giữa nhóm. Không chọn case theo nội dung để làm đẹp giả thuyết.
Population: 97 selection_loss, 17 regression complete-hit, 233 success.
Regression ở đây định nghĩa chặt: reranker complete-evidence@10=0 và RRF pool50=1;
không đổi qua Recall/MRR sau khi xem dữ liệu. Đây là mẫu có chủ đích phân tầng,
không phải ước lượng tỷ lệ lỗi toàn diagnosis; các query cùng paper có tương quan.

QASPER thành công khi đủ **một** annotation text-only hợp lệ, không cần toàn bộ
union annotations. Unanswerable/figure-table/unmatched/empty annotation bị loại.
So sánh evidence theo (paper, NFC + chuẩn hóa whitespace text); hit trùng vẫn
chiếm rank. Từng alternative và phần còn thiếu được giữ riêng trong packet.
Không coi gold tùy chọn của alternative khác là lỗi nếu đã đủ một alternative.

- Trùng chính xác: hai hit có normalized text tương đương trong cùng paper.
- High overlap: word-token set Jaccard ≥0.8, cả hai có ≥20 unique word tokens;
  đây là phép đo bề mặt, không phải semantic redundancy. Không hạ ngưỡng sau review.
- Lân cận: cùng paper, cùng numeric section_index, paragraph_index chênh đúng 1.
  Không suy vị trí từ ID; title/abstract hoặc index null không được coi là đoạn cạnh.
- `one_hop_available`: có một alternative chưa đủ mà mọi unit còn thiếu đều có
  bản tương đương trong pool nằm cạnh một đoạn reranker đã chọn. Đây là kiểm tra
  **dùng gold, cho phép thêm không giới hạn**, chưa áp replacement/token budget;
  không phải hiệu quả selector, không phải số case pilot chắc chắn sửa được.
- Adjacency links có thể lặp qua các alternative annotations; số case dùng unique
  query, không dùng số dòng link làm denominator.

## Quan sát tự động

| Nhóm | Query | Có exact duplicate top10 | Có high-overlap pair | Có gold thiếu nằm cạnh hit đã chọn | Có alternative đủ nếu thêm các neighbor cần thiết |
|---|---:|---:|---:|---:|---:|

| Selection loss | 20 | 0 | 0 | 8 | 5 |
| Regression complete-hit | 10 | 0 | 0 | 5 | 5 |
| Success control | 10 | 0 | 0 | 0 | 0 |
| Unique | 37 | 0 | 0 | 11 | 8 |

Trong 27 failure query duy nhất: 11 có ít nhất một gold thiếu nằm cạnh hit đã chọn;
8 có thể hoàn tất một alternative trong phép kiểm tra thêm-neighbor không giới hạn.
16 failure không có link adjacency như vậy; không được mặc định cùng một cách sửa.
Success có 0 “missing-gold adjacency” không phải bằng chứng context vô ích ở success:
đây là phép kiểm tra unit thiếu, không kiểm tra mọi liên hệ ngữ cảnh trong top10.
Không thấy duplicate theo định nghĩa này cũng không bác bỏ semantic redundancy.

## Trợ lý đọc nội dung: bằng chứng ủng hộ và phản chứng

Các nhận xét sau dựa trên text, ID và rank đã hiển thị trong packet; **chưa được
người xác minh**. Chúng kiểm tra giả thuyết, không gán nguyên nhân model.

### Ví dụ có hỗ trợ cho tín hiệu đoạn lân cận

1. [Q `af34051bf3e628c1e2a00b110bb84e5f018b419f`](../analysis/qasper_review_v1/cases/af34051bf3e628c1e2a00b110bb84e5f018b419f.md),
   “What are the baselines?” Reranker đã giữ `qasper:1909.07575:s13:p0–p3`
   và `s13:p5`, nhưng bỏ `s13:p4`. Đoạn p4 mở đầu **“Many-to-many+pre-training”**
   và mô tả **“Triangle+pre-train”**, nối tiếp danh sách baseline ở p1–p3;
   p3 rank1 và p5 rank5 kẹp hai phía. Đây là evidence nội dung cụ thể cho list
   completion, không phải duplicate. Recall 0.8 vẫn chưa complete; RRF recall0.6
   cũng chưa complete. Một neighbor cần thiết trong alternative này nằm ngay trong pool.

2. [Q `f8f4e4a50d2b3fbd193327e79ea32d8d057e1414`](../analysis/qasper_review_v1/cases/f8f4e4a50d2b3fbd193327e79ea32d8d057e1414.md),
   “How was the dataset collected?” `qasper:2002.01320:s1:p1` ở rank6 nói tái sử
   dụng validated CoVo và split; gold `s1:p0` ngay trước giải thích contributors
   **record voice clips** và mỗi clip được **validated by at least two other users**.
   Tương tự `s2:p1` ở rank4 mô tả xây evaluation TT, còn gold `s2:p0` mô tả
   nguồn Tatoeba. RRF giữ đủ một alternative (complete1), reranker complete0.
   Thông tin nguồn/thu thập bị chia đoạn là giả thuyết được text hỗ trợ; vẫn cần
   kiểm tra chi phí và đoạn bị thay thế, không suy ra mọi dataset question đều giống vậy.

3. [Q `c5abe97625b9e1c8de8208e15d59c704a597b88c`](../analysis/qasper_review_v1/cases/c5abe97625b9e1c8de8208e15d59c704a597b88c.md),
   “What are the results from these proposed strategies?” Gold
   `qasper:2002.08795:s2:p5` nêu **score of 40** và việc vượt bottleneck;
   reranker giữ p4 rank7 (định nghĩa A2C-Explore) và p6 rank6 (bàn key insights)
   nhưng bỏ p5 nằm giữa. Hai đầu context có trong top10 và kết quả định lượng
   ở giữa có trong pool. Cả reranker và RRF đều recall0; không chỉ là regression.

4. [Q `d650101712e36594bd77b45930a990402a455222`](../analysis/qasper_review_v1/cases/d650101712e36594bd77b45930a990402a455222.md),
   “what datasets are used in the experiment?” `qasper:1809.06537:s11:p1`
   rank2 chỉ nói preprocessing; gold p0 ngay trước nêu **China Judgments Online**,
   thu thập cases và chia train/validation/test. RRF complete1, reranker0.
   Đây là nguồn corpus cụ thể nằm cạnh đoạn thủ tục, không chỉ liên hệ qua keyword.

5. [Q `455d4ef8611f62b1361be4f6387b222858bb5e56`](../analysis/qasper_review_v1/cases/455d4ef8611f62b1361be4f6387b222858bb5e56.md),
   “How was this data collected?” `qasper:1603.09631:s4:p0` rank2 mô tả
   **“A perfect data collection scenario”**; gold p1 mở **“However”** và nêu
   **CrowdFlower (CF)** là cách làm thực tế. Cặp contrast nối tiếp này hỗ trợ
   đọc ngữ cảnh; title “Dataset Collection Process” rank8 chưa thay thế p1.
   Không bắt buộc lấy cả union gold p1+p2 nếu một annotation hợp lệ đã đủ.

### Phản chứng và giới hạn quan trọng

6. [Q `06b5272774ec43ee5facfa7111033386f06cf448`](../analysis/qasper_review_v1/cases/06b5272774ec43ee5facfa7111033386f06cf448.md),
   “Is an instance a sentence or an IE tuple?” Top10 đã có
   `qasper:1905.07791:s2:p2` rank10, nói rõ **“analysis at the sentence level”**;
   gold p1 mô tả difficult/easy examples và label trong bảng. Mặc dù gold nằm
   cạnh hit, không thể kết luận reranker mất câu trả lời semantic: đoạn đã chọn
   có thể trực tiếp trả lời hơn gold annotation. Cần người kiểm tra annotation
   alignment; không gán “context failure” chỉ từ complete-hit0.

7. [Q `7b4fb6da74e6bd1baea556788a02969134cf0800`](../analysis/qasper_review_v1/cases/7b4fb6da74e6bd1baea556788a02969134cf0800.md),
   “Do they train their model starting from a checkpoint?” Gold
   `qasper:1909.03405:s6:p0` mô tả BERTbase layers/heads; top10 p1 rank5 mô tả
   training phases/corpus. p0 liền trước nên tăng annotation completeness là khả
   dĩ, nhưng p0 không trực tiếp xác nhận checkpoint initialization. Đây là case
   cần kiểm tra ý nghĩa annotation, không là bằng chứng causal cho selector.

8. [Q `2c947447d81252397839d58c75ebcc71b34379b5`](../analysis/qasper_review_v1/cases/2c947447d81252397839d58c75ebcc71b34379b5.md),
   “On what Text-Based Games are experiments performed?” Top10
   `qasper:2001.08868:s6:p4` rank5 đã nêu **CoinCollector** và **CookingWorld**;
   gold p2+p3 định nghĩa từng game. p3 nằm cạnh p4 nhưng p2 cách hai đoạn:
   một bước neighbor không hoàn tất alternative. Đây vừa là giới hạn coverage
   vừa cho thấy câu trả lời có thể có trong đoạn chưa annotate; cần human review.

9. [Q `48088a842f7a433d3290eb45eb0d4c6ab1d8f13c`](../analysis/qasper_review_v1/cases/48088a842f7a433d3290eb45eb0d4c6ab1d8f13c.md),
   “What learning models are used on the dataset?” `qasper:1808.10245:s3:p0`
   rank10 mở danh sách năm classifiers; gold p1 **Naïve Bayes**, p2–p5 là các
   mục khác, và `s4:p1` mô tả CNN nằm cạnh `s4:p0` rank1. Một neighbor đơn lẻ
   không đủ cả alternative 7 units; không được xem adjacency là sửa toàn bộ lỗi.

10. [Q `0752d71a0a1f73b3482a888313622ce9e9870d6e`](../analysis/qasper_review_v1/cases/0752d71a0a1f73b3482a888313622ce9e9870d6e.md),
    “what was the baseline?” Rank1 là `qasper:1708.01464:s9:title` (“Baseline”),
    gold `s9:p0` gọi tên **wFST**. Có liên hệ heading–paragraph nhưng predicate
    numeric paragraph adjacency không áp cho title. Top2/top3 cũng nhắc baseline/
    wFST; không gán lexical distractor hoặc mở rộng heuristic để “sửa” mẫu này.

11. [Q `07d15501a599bae7eb4a9ead63e9df3d55b3dc35`](../analysis/qasper_review_v1/cases/07d15501a599bae7eb4a9ead63e9df3d55b3dc35.md),
    “How do they obtain psychological dimensions of people?” Top đầu mô tả
    vocabulary/blog maps; gold `qasper:1612.06685:s5:p3` nêu **Meaning Extraction
    Method (MEM)**. Không có gold-neighbor link với top10. Pilot adjacency không
    mặc nhiên xử lý case này; chưa suy nguyên nhân ranking từ text tương đồng.

### Đối chứng success và preservation risk

12. [Q `4625cfba3083346a96e573af5464bc26c34ec943`](../analysis/qasper_review_v1/cases/4625cfba3083346a96e573af5464bc26c34ec943.md),
    “by how much did their model improve?” Reranker giữ hai gold liên tiếp
    `qasper:1810.04428:s5:p6` rank1 (2.11 BLEU, 1.7 FKGL, 1.07 SARI) và
    p7 rank2 (6.37 BLEU). Đây là ví dụ context bổ trợ đã được giữ tốt;
    không được thay top10 một cách mù quáng làm mất unit khác.

13. [Q `0810b43404686ddfe4ca84783477ae300fdd2ea4`](../analysis/qasper_review_v1/cases/0810b43404686ddfe4ca84783477ae300fdd2ea4.md),
    “On top of BERT does the RNN layer work better or the transformer layer?”
    Packet có gold `qasper:1910.10781:s0:p2`, `s11:p0`, `s12:p0` thuộc các
    alternative. Reranker complete1 dù thiếu `s12:p0` của alternative khác.
    Không “sửa” success chỉ vì union gold còn thiếu; các summary/results không
    liên tiếp có thể cùng hữu ích. Giữ semantics annotation khi đánh giá pilot.

## Một thí nghiệm đề xuất, chưa chạy

**Neighbor-one-swap**: so sánh top10 reranker gốc với đúng một luật thay thế
không dùng gold. Giữ nguyên candidate pool, source/model SHA và text units:

1. Từ các paragraph đã ở top10, xét paragraph chưa chọn nằm ngay trước/sau
   trong cùng numeric section, chỉ khi candidate đó vốn thuộc union top50.
2. Xếp các neighbor theo RRF k=60 tính từ BM25/dense rank đã lưu, tie evidence ID.
   Không dùng gold, không dùng `one_hop_available`, không dùng label review hoặc
   reranker score giả cho candidate ngoài top10. Bỏ ID/text-equivalent đã có.
3. Chọn tối đa một neighbor đầu tiên thỏa chi phí; thay đúng item rank10 của
   reranker, hoặc giữ nguyên nếu không có candidate hợp lệ. Không thêm đoạn thứ11,
   không ghép/expand text, không thêm title rule/2-hop/penalty trùng vào ablation này.
4. Kiểm tra chi phí candidate ≤ item bị thay bằng **cùng tokenizer và cùng cách
   tính context**; khi đó số evidence không đổi và tổng token không tăng so với
   baseline của query. Chưa có token costs thì chưa chạy/tuyên bố matched-token.
5. Freeze luật trên trước khi chạy. Báo paired complete-hit/Recall@10/MRR@10,
   win/loss và số success bị hỏng; QASPER bootstrap theo paper nếu đánh giá.
   Không tune trên mẫu37 hoặc mở reporting; không coi gold-assisted8/27 là gain.

Text ở các case1–5 hỗ trợ thử tín hiệu lân cận trong một pilot có kiểm soát;
case6–13 giới hạn kỳ vọng và yêu cầu kiểm tra regression/annotation. Đây không
phải kết luận context là nguyên nhân chính hay selector sẽ tốt hơn.

**Dữ liệu/score còn thiếu:**

- Cross-encoder chỉ lưu logits top10; ngoài top10 không có logits hoặc pair-token
  length riêng. Pilot trên cố ý chỉ dùng top10 sẵn có và source ranks nên **không
  cần** suy luận lại model để lấy full-union logits. Không thể làm score-margin/
  learned context reranking từ artifact hiện tại; các biến thể đó không đề xuất ở đây.
- Chưa có tokenizer-specific context cost cho từng unit hoặc reader/context-token
  budget đã chốt. Hiện ngân sách chắc chắn chỉ là top10 evidence, max-length512
  của cross-encoder không phải reader context budget. Cần chốt tokenizer/cost rule,
  tính baseline/candidate costs và lưu hash trước khi thử replacement với token cap.
- Metadata paper/section/paragraph, full pool texts, BM25/dense ranks và top10 hai
  phương pháp đã có. Human-verified labels vẫn thiếu, đặc biệt annotation ambiguity
  ở case6–8. Không tạo dữ liệu calibration/reporting để bù phần thiếu.

## Danh sách case và packet đầy đủ

Mỗi liên kết mở một packet chứa question, gold text theo alternative, top10 hai
method và full candidate pool. Bảng sau là facts tự động; nhãn người vẫn trống.

| Query ID / full packet | Nhóm | Question | Recall reranker / RRF | Gold thiếu cạnh hit | One-hop alternative (oracle) | Human |
|---|---|---|---:|---|---|---|

| [qasper:06b5272774ec43ee5facfa7111033386f06cf448](../analysis/qasper_review_v1/cases/06b5272774ec43ee5facfa7111033386f06cf448.md) | regression_complete | Is an instance a sentence or an IE tuple? | 0.0000 / 1.0000 | True | True | PENDING |
| [qasper:0752d71a0a1f73b3482a888313622ce9e9870d6e](../analysis/qasper_review_v1/cases/0752d71a0a1f73b3482a888313622ce9e9870d6e.md) | selection_loss | what was the baseline? | 0.0000 / 0.0000 | False | False | PENDING |
| [qasper:07d15501a599bae7eb4a9ead63e9df3d55b3dc35](../analysis/qasper_review_v1/cases/07d15501a599bae7eb4a9ead63e9df3d55b3dc35.md) | selection_loss;regression_complete | How do they obtain psychological dimensions of people? | 0.0000 / 1.0000 | False | False | PENDING |
| [qasper:0810b43404686ddfe4ca84783477ae300fdd2ea4](../analysis/qasper_review_v1/cases/0810b43404686ddfe4ca84783477ae300fdd2ea4.md) | success | On top of BERT does the RNN layer work better or the transformer layer? | 1.0000 / 0.5000 | False | False | PENDING |
| [qasper:1ed49a8c07ef0ac15cfa6b7decbde6604decbd5b](../analysis/qasper_review_v1/cases/1ed49a8c07ef0ac15cfa6b7decbde6604decbd5b.md) | selection_loss;regression_complete | What dataset does this approach achieve state of the art results on? | 0.5000 / 1.0000 | True | True | PENDING |
| [qasper:21548433abd21346659505296fb0576e78287a74](../analysis/qasper_review_v1/cases/21548433abd21346659505296fb0576e78287a74.md) | success | What dataset is used for this work? | 1.0000 / 1.0000 | False | False | PENDING |
| [qasper:2317ca8d475b01f6632537b95895608dc40c4415](../analysis/qasper_review_v1/cases/2317ca8d475b01f6632537b95895608dc40c4415.md) | success | How is a "chunk of posts" defined in this work? | 1.0000 / 1.0000 | False | False | PENDING |
| [qasper:23b2901264bda91045258b5d4120879ae292e950](../analysis/qasper_review_v1/cases/23b2901264bda91045258b5d4120879ae292e950.md) | selection_loss | What are method improvements of F1 for paraphrase identification? | 0.6667 / 0.6667 | False | False | PENDING |
| [qasper:2740e3d7d33173664c1c5ab292c7ec75ff6e0802](../analysis/qasper_review_v1/cases/2740e3d7d33173664c1c5ab292c7ec75ff6e0802.md) | selection_loss | what datasets were used? | 0.0000 / 0.5000 | False | False | PENDING |
| [qasper:2ad4d3d222f5237ed97923640bc8e199409cbe52](../analysis/qasper_review_v1/cases/2ad4d3d222f5237ed97923640bc8e199409cbe52.md) | success | What user variations have been tested? | 1.0000 / 1.0000 | False | False | PENDING |
| [qasper:2c6b50877133a499502feb79a682f4023ddab63e](../analysis/qasper_review_v1/cases/2c6b50877133a499502feb79a682f4023ddab63e.md) | regression_complete | what language does this paper focus on? | 0.0000 / 1.0000 | False | False | PENDING |
| [qasper:2c947447d81252397839d58c75ebcc71b34379b5](../analysis/qasper_review_v1/cases/2c947447d81252397839d58c75ebcc71b34379b5.md) | selection_loss | On what Text-Based Games are experiments performed? | 0.0000 / 0.0000 | True | False | PENDING |
| [qasper:31d695ba855d821d3e5cdb7bea638c7dbb7c87c7](../analysis/qasper_review_v1/cases/31d695ba855d821d3e5cdb7bea638c7dbb7c87c7.md) | regression_complete | Which neural architecture do they use as a base for their attention conflict mechanisms? | 0.0000 / 1.0000 | False | False | PENDING |
| [qasper:42084c41343e5a6ae58a22e5bfc5ce987b5173de](../analysis/qasper_review_v1/cases/42084c41343e5a6ae58a22e5bfc5ce987b5173de.md) | selection_loss | Does the paper explore extraction from electronic health records? | 0.0000 / 0.0000 | False | False | PENDING |
| [qasper:455d4ef8611f62b1361be4f6387b222858bb5e56](../analysis/qasper_review_v1/cases/455d4ef8611f62b1361be4f6387b222858bb5e56.md) | selection_loss | How was this data collected? | 0.0000 / 1.0000 | True | True | PENDING |
| [qasper:4625cfba3083346a96e573af5464bc26c34ec943](../analysis/qasper_review_v1/cases/4625cfba3083346a96e573af5464bc26c34ec943.md) | success | by how much did their model improve? | 1.0000 / 0.5000 | False | False | PENDING |
| [qasper:48088a842f7a433d3290eb45eb0d4c6ab1d8f13c](../analysis/qasper_review_v1/cases/48088a842f7a433d3290eb45eb0d4c6ab1d8f13c.md) | selection_loss | What learning models are used on the dataset? | 0.2857 / 0.1429 | True | False | PENDING |
| [qasper:498c0229f831c82a5eb494cdb3547452112a66a0](../analysis/qasper_review_v1/cases/498c0229f831c82a5eb494cdb3547452112a66a0.md) | selection_loss | How do they match annotators to instances? | 0.0000 / 1.0000 | False | False | PENDING |
| [qasper:4eaf9787f51cd7cdc45eb85cf223d752328c6ee4](../analysis/qasper_review_v1/cases/4eaf9787f51cd7cdc45eb85cf223d752328c6ee4.md) | success | what datasets did they use? | 1.0000 / 1.0000 | False | False | PENDING |
| [qasper:521280a87c43fcdf9f577da235e7072a23f0673e](../analysis/qasper_review_v1/cases/521280a87c43fcdf9f577da235e7072a23f0673e.md) | success | How many annotators participated? | 1.0000 / 1.0000 | False | False | PENDING |
| [qasper:568fb7989a133564d84911e7cb58e4d8748243ef](../analysis/qasper_review_v1/cases/568fb7989a133564d84911e7cb58e4d8748243ef.md) | selection_loss | How is trajectory with how rewards extracted? | 0.0000 / 1.0000 | False | False | PENDING |
| [qasper:7b4fb6da74e6bd1baea556788a02969134cf0800](../analysis/qasper_review_v1/cases/7b4fb6da74e6bd1baea556788a02969134cf0800.md) | regression_complete | Do they train their model starting from a checkpoint? | 0.5000 / 1.0000 | True | True | PENDING |
| [qasper:8c48c726bb17a17d70ab29db4d65a93030dd5382](../analysis/qasper_review_v1/cases/8c48c726bb17a17d70ab29db4d65a93030dd5382.md) | selection_loss | How much data is needed to train the task-specific encoder? | 0.5000 / 0.5000 | False | False | PENDING |
| [qasper:8d4ac4afbf5b14f412171729ceb5e822afcfa3f4](../analysis/qasper_review_v1/cases/8d4ac4afbf5b14f412171729ceb5e822afcfa3f4.md) | regression_complete | Do they build a model to automatically detect demographic, lingustic or psycological dimensons of people? | 0.0000 / 1.0000 | False | False | PENDING |
| [qasper:af34051bf3e628c1e2a00b110bb84e5f018b419f](../analysis/qasper_review_v1/cases/af34051bf3e628c1e2a00b110bb84e5f018b419f.md) | selection_loss | What are the baselines? | 0.8000 / 0.6000 | True | True | PENDING |
| [qasper:b1cf5739467ba90059add58d11b73d075a11ec86](../analysis/qasper_review_v1/cases/b1cf5739467ba90059add58d11b73d075a11ec86.md) | regression_complete | Do they report results only on English? | 0.5000 / 1.0000 | False | False | PENDING |
| [qasper:b249b60a8c94d0e40d65f1ffdfcac527dab57516](../analysis/qasper_review_v1/cases/b249b60a8c94d0e40d65f1ffdfcac527dab57516.md) | success | Do the hashtag and SemEval datasets contain only English data? | 1.0000 / 1.0000 | False | False | PENDING |
| [qasper:ba539cab80d25c3e20f39644415ed48b9e4e4185](../analysis/qasper_review_v1/cases/ba539cab80d25c3e20f39644415ed48b9e4e4185.md) | selection_loss | How do the backoff strategies work? | 0.5000 / 0.5000 | True | False | PENDING |
| [qasper:beac555c4aea76c88f19db7cc901fa638765c250](../analysis/qasper_review_v1/cases/beac555c4aea76c88f19db7cc901fa638765c250.md) | selection_loss | What useful information does attention capture? | 0.5000 / 0.5000 | False | False | PENDING |
| [qasper:c5abe97625b9e1c8de8208e15d59c704a597b88c](../analysis/qasper_review_v1/cases/c5abe97625b9e1c8de8208e15d59c704a597b88c.md) | selection_loss | What are the results from these proposed strategies? | 0.0000 / 0.0000 | True | True | PENDING |
| [qasper:c7eb71683f53ab7acffd691a36cad6edc7f5522e](../analysis/qasper_review_v1/cases/c7eb71683f53ab7acffd691a36cad6edc7f5522e.md) | success | Do they report results only on English data? | 1.0000 / 0.5000 | False | False | PENDING |
| [qasper:c88a846197b72d25e04ec55f00ee3e72f655504c](../analysis/qasper_review_v1/cases/c88a846197b72d25e04ec55f00ee3e72f655504c.md) | success | Which dataset do they use? | 1.0000 / 1.0000 | False | False | PENDING |
| [qasper:cb77d6a74065cb05318faf57e7ceca05e126a80d](../analysis/qasper_review_v1/cases/cb77d6a74065cb05318faf57e7ceca05e126a80d.md) | selection_loss | What is the baseline? | 0.0000 / 0.0000 | False | False | PENDING |
| [qasper:d3aa0449708cc861a51551b128d73e11d62207d2](../analysis/qasper_review_v1/cases/d3aa0449708cc861a51551b128d73e11d62207d2.md) | selection_loss | What they use in their propsoed framework? | 0.0000 / 0.0000 | False | False | PENDING |
| [qasper:d650101712e36594bd77b45930a990402a455222](../analysis/qasper_review_v1/cases/d650101712e36594bd77b45930a990402a455222.md) | regression_complete | what datasets are used in the experiment? | 0.0000 / 1.0000 | True | True | PENDING |
| [qasper:dac087e1328e65ca08f66d8b5307d6624bf3943f](../analysis/qasper_review_v1/cases/dac087e1328e65ca08f66d8b5307d6624bf3943f.md) | selection_loss | LDA is an unsupervised method; is this paper introducing an unsupervised approach to spam detection? | 0.0000 / 0.0000 | False | False | PENDING |
| [qasper:f8f4e4a50d2b3fbd193327e79ea32d8d057e1414](../analysis/qasper_review_v1/cases/f8f4e4a50d2b3fbd193327e79ea32d8d057e1414.md) | selection_loss;regression_complete | How was the dataset collected? | 0.5000 / 1.0000 | True | True | PENDING |

## Provenance, tái lập và kiểm tra

- Selection/provenance: [selection.json](../analysis/qasper_review_v1/selection.json).
- Facts từng query: [observations.csv](../analysis/qasper_review_v1/observations.csv).
- Hash từng packet/output: [output_hashes.json](../analysis/qasper_review_v1/output_hashes.json).
- ZIP SHA256: `a2aa11be3c93861739d79ebceb3219b3dc50d8b95d1b06ac93aaeb6011051bc0`;
  kiểm tra lại sidecar, mọi ZIP member và baseline output hashes trước khi chọn mẫu.
- Chỉ đọc gold từ diagnosis exports và kiểm tra membership với manifest QASPER;
  không decode full-gold JSONL, không chấm per-query calibration/reporting.
- Output mới dùng exclusive creation; không sửa artifact/manifest/notebook cũ.
- 10 tests pass (5 review, 5 baseline guards): seed/overlap/regression definition,
  adjacency metadata, alternative/optional gold, ngoài-pool không thành candidate,
  multi-evidence completion, duplicate normalization, denominator/hash/leakage guards.

```powershell
.\.venv\Scripts\python.exe -m unittest scripts.test_prepare_qasper_review scripts.test_analyze_baseline_diagnosis -v
# qasper_review_v1 đã có. Nếu cần tái lập, dùng output-dir mới:
.\.venv\Scripts\python.exe -m scripts.prepare_qasper_review --archive artifacts/artifacts_VERIFIED_COMPLETE_20261031T170355811993Z_1bcc0097.zip --sha256-file artifacts/artifacts_VERIFIED_COMPLETE_20261031T170355811993Z_1bcc0097.sha256.json --output-dir analysis/qasper_review_v2 --seed 42
```

## File thay đổi trong đợt review này

Danh sách chính xác 45 file; không gồm các thay đổi `.gitignore`, artifact hoặc
output baseline đã tồn tại trước đợt này. Không commit/push tự động.

```text
docs/MANUAL_REVIEW.md
docs/PROGRESS.md
docs/NEXT_STEPS.md
scripts/prepare_qasper_review.py
scripts/test_prepare_qasper_review.py
analysis/qasper_review_v1/cases/06b5272774ec43ee5facfa7111033386f06cf448.md
analysis/qasper_review_v1/cases/0752d71a0a1f73b3482a888313622ce9e9870d6e.md
analysis/qasper_review_v1/cases/07d15501a599bae7eb4a9ead63e9df3d55b3dc35.md
analysis/qasper_review_v1/cases/0810b43404686ddfe4ca84783477ae300fdd2ea4.md
analysis/qasper_review_v1/cases/1ed49a8c07ef0ac15cfa6b7decbde6604decbd5b.md
analysis/qasper_review_v1/cases/21548433abd21346659505296fb0576e78287a74.md
analysis/qasper_review_v1/cases/2317ca8d475b01f6632537b95895608dc40c4415.md
analysis/qasper_review_v1/cases/23b2901264bda91045258b5d4120879ae292e950.md
analysis/qasper_review_v1/cases/2740e3d7d33173664c1c5ab292c7ec75ff6e0802.md
analysis/qasper_review_v1/cases/2ad4d3d222f5237ed97923640bc8e199409cbe52.md
analysis/qasper_review_v1/cases/2c6b50877133a499502feb79a682f4023ddab63e.md
analysis/qasper_review_v1/cases/2c947447d81252397839d58c75ebcc71b34379b5.md
analysis/qasper_review_v1/cases/31d695ba855d821d3e5cdb7bea638c7dbb7c87c7.md
analysis/qasper_review_v1/cases/42084c41343e5a6ae58a22e5bfc5ce987b5173de.md
analysis/qasper_review_v1/cases/455d4ef8611f62b1361be4f6387b222858bb5e56.md
analysis/qasper_review_v1/cases/4625cfba3083346a96e573af5464bc26c34ec943.md
analysis/qasper_review_v1/cases/48088a842f7a433d3290eb45eb0d4c6ab1d8f13c.md
analysis/qasper_review_v1/cases/498c0229f831c82a5eb494cdb3547452112a66a0.md
analysis/qasper_review_v1/cases/4eaf9787f51cd7cdc45eb85cf223d752328c6ee4.md
analysis/qasper_review_v1/cases/521280a87c43fcdf9f577da235e7072a23f0673e.md
analysis/qasper_review_v1/cases/568fb7989a133564d84911e7cb58e4d8748243ef.md
analysis/qasper_review_v1/cases/7b4fb6da74e6bd1baea556788a02969134cf0800.md
analysis/qasper_review_v1/cases/8c48c726bb17a17d70ab29db4d65a93030dd5382.md
analysis/qasper_review_v1/cases/8d4ac4afbf5b14f412171729ceb5e822afcfa3f4.md
analysis/qasper_review_v1/cases/af34051bf3e628c1e2a00b110bb84e5f018b419f.md
analysis/qasper_review_v1/cases/b1cf5739467ba90059add58d11b73d075a11ec86.md
analysis/qasper_review_v1/cases/b249b60a8c94d0e40d65f1ffdfcac527dab57516.md
analysis/qasper_review_v1/cases/ba539cab80d25c3e20f39644415ed48b9e4e4185.md
analysis/qasper_review_v1/cases/beac555c4aea76c88f19db7cc901fa638765c250.md
analysis/qasper_review_v1/cases/c5abe97625b9e1c8de8208e15d59c704a597b88c.md
analysis/qasper_review_v1/cases/c7eb71683f53ab7acffd691a36cad6edc7f5522e.md
analysis/qasper_review_v1/cases/c88a846197b72d25e04ec55f00ee3e72f655504c.md
analysis/qasper_review_v1/cases/cb77d6a74065cb05318faf57e7ceca05e126a80d.md
analysis/qasper_review_v1/cases/d3aa0449708cc861a51551b128d73e11d62207d2.md
analysis/qasper_review_v1/cases/d650101712e36594bd77b45930a990402a455222.md
analysis/qasper_review_v1/cases/dac087e1328e65ca08f66d8b5307d6624bf3943f.md
analysis/qasper_review_v1/cases/f8f4e4a50d2b3fbd193327e79ea32d8d057e1414.md
analysis/qasper_review_v1/observations.csv
analysis/qasper_review_v1/output_hashes.json
analysis/qasper_review_v1/selection.json
```
