# Lab 25 — GPU FinOps: Bài viết ngắn

**Sinh viên:** Đoàn Quốc Việt · **Mã:** 2A202601623 · **Track 2 (Infrastructure) · Day 25**
**Đầu ra kèm theo:** `outputs/report.md`, `outputs/savings.png`, `outputs/focus_export.csv`, `outputs/missions_output.txt`

Tất cả con số dưới đây được sinh trực tiếp từ `python missions/run_all.py` trên dữ liệu seed=25 (tái tạo được 100%).

---

## 1. Baseline vs. Optimized

| Chỉ số | Baseline | Optimized | Thay đổi |
|---|---|---|---|
| Chi tiêu GPU / tháng | **$27,133** | **$14,223** | **−$12,910 (−47.6%)** |
| Đơn giá inference | **$6.488 / 1M-token** | **$1.126 / 1M-token** | **−82.6%** |
| Chi phí inference / ngày | $48.87 | $8.48 | −$40.39 |

Baseline = deployment ngây thơ: mọi request chạy model lớn, không cache, không batch, và toàn bộ 8 workload mua on-demand 100%.

| Đòn bẩy | Tiết kiệm / tháng | Tỷ trọng |
|---|---|---|
| Purchasing (spot / reserved) | $9,788 | 75.8% |
| Right-size các GPU "util-lie" | $1,310 | 10.1% |
| Inference (cascade + cache + batch) | $1,212 | 9.4% |
| Tắt GPU idle | $600 | 4.6% |

> Lưu ý về đơn vị: tính theo **$/GPU-giờ** thì purchasing là đòn bẩy lớn nhất; nhưng tính theo **$/1M-token** thì inference giảm tới 82.6% — đây chính là ví dụ cho câu hỏi "khi nào hai đơn vị cho kết quả trái ngược". Purchasing hạ *giá thuê*, inference hạ *lượng tài nguyên cần thuê*. Đòn bẩy thứ hai mới là thứ scale theo tăng trưởng người dùng.

## 2. Đòn bẩy nào đóng góp nhiều nhất và tại sao

Trong nội bộ inference, giá trị biên (leave-one-out, tính ngược từ hoá đơn đã tối ưu):

| Đòn bẩy | $/ngày | $/tháng | Cơ chế |
|---|---|---|---|
| Cascade (routing sang model nhỏ) | $27.64 | $829 | 80% traffic là request dễ; tier nhỏ rẻ hơn ~15× mỗi token |
| Batch API | $1.79 | $54 | −50% cho traffic chấp nhận hàng đợi (toàn bộ traffic của team `eval`) |
| Prompt caching | $1.17 | $35 | −90% cho phần input đã cache; chỉ chat/RAG có system prompt dùng chung lớn |

**Cascade lớn hơn hai đòn bẩy còn lại cộng lại gần 10 lần.** Lý do bản chất: batch và cache chỉ *nhân một hệ số chiết khấu* lên giá token (0.5 và 0.1), còn cascade *đổi hẳn loại token mình mua* — từ $3.00/$15.00 xuống $0.20/$0.40 per 1M. Chiết khấu bị chặn trên bởi mức nhà cung cấp cho; routing thì không.

Ở tầng hạ tầng, purchasing lớn nhất vì đơn giản là nó tác động lên khoản chi lớn nhất ($25,667/tháng on-demand): 5/8 job là interruptible và chạy được trên spot với checkpoint, 3 job serving 24/7 thì cam kết reserved.

## 3. GPU-Util Lie

| GPU | Loại | GPU-Util | MFU | MBU | Bị tính tiền/tháng | Đốt vô ích/tháng |
|---|---|---|---|---|---|---|
| `gpu-h100-4` | H100 | 98.2% | 0.19 | 0.21 | $1,800 | **$1,451** |
| `gpu-a10g-1` | A10G | 96.9% | 0.27 | 0.30 | $720 | **$527** |

**Cơ chế:** `nvidia-smi` GPU-Util chỉ trả lời một câu hỏi — *trong cửa sổ lấy mẫu, có ít nhất một kernel đang nằm trên device hay không?* Nó là **bộ đếm duty-cycle, không phải bộ đếm throughput**. Một kernel dành phần lớn thời gian *stall* chờ đọc HBM, hoặc một chuỗi kernel bé xíu mà chi phí launch lớn hơn phần toán học, vẫn giữ bộ đếm đó ở ~100% trong khi tensor core đứng yên. Roofline xác nhận: `gpu-h100-4` chạy ở ~278 FLOP/byte so với ridge point 296 FLOP/byte của H100 → memory-bound. Ta thuê 990 TFLOP/s và nhận về ~190.

**Hệ quả tài chính (phần nguy hiểm nhất):** dashboard dựa trên util báo GPU này "khoẻ và đã dùng hết công suất", nên (a) không ai right-size nó, và (b) capacity planning đi mua thêm đúng loại SKU đó. Tổng cộng $1,978/tháng bị đốt trên 2 GPU, và sai lầm này *tự nhân bản* qua mỗi vòng mua sắm. Cách chặn: đo MFU/MBU theo job, cảnh báo khi `util > 90% AND MFU < 0.30`.

## 4. Các phần mở rộng đã thực hiện (5/5)

| # | Extension | File | Kết quả đo được |
|---|---|---|---|
| 1 | `recommend_tier_v2()` — nhận biết rủi ro & kỳ hạn | `finops/pricing.py`, `missions/m3_purchasing.py` | v1 báo tiết kiệm **39.1%**, v2 báo **38.1%** ($15,879/mo). Chênh lệch không phải do v2 kém: v1 tính reserved theo **giờ dùng thật** (540h cho `job-infer-search` 18h/ngày), trong khi reserved luôn bị tính đủ **720h cam kết** → v1 tạo ra $324/tháng "tiết kiệm ảo". v2 còn dùng interruption rate riêng theo GPU (H100 3%/h ↔ L4 15%/h) và chỉ cho ký 3yr khi job chạy ≥28 ngày. **Báo cáo M5 dùng số của v2.** |
| 2 | Right-sizing theo roofline / MBU | `missions/m1_efficiency_audit.py` | **$1,310/tháng (8.5% hoá đơn fleet)**. Xếp hạng catalog theo `$/TB-s` và `$/GB-VRAM`: L4 rẻ nhất theo $/GPU-giờ ($0.80) nhưng **đắt nhất theo băng thông** ($2.667/TB-s so với $0.746 của H100). Chỉ hạ cấp GPU vừa memory-bound vừa MFU < 0.35 (h100-4, h100-5 → A100; a10g-0, a10g-1 → L4), và chỉ khi SKU mới vẫn đủ băng thông + VRAM đo được cộng 15% headroom. |
| 3 | `cache_is_worth_it()` | `finops/pricing.py`, `missions/m2_inference_levers.py` | Break-even: tier **small cần 5.8 lượt đọc**, tier **large chỉ cần 0.6** — vì mỗi lượt đọc tiết kiệm 90% *giá token*, còn tiền thuê lưu trữ cache thì không rẻ đi theo model. Dataset thực tế: 237.8 lượt đọc/prefix (small) và 62.2 (large) → dư **40.8×** và **96×** ngưỡng, nên lever cache được áp dụng cho cả hai tier. |
| 4 | Ngân sách Reasoning | `missions/m2_inference_levers.py` | Reasoning = **8.4% traffic → 16.5% chi phí → 94.0% năng lượng** (148.20 Wh vs 0.86 Wh mỗi request). Cap 10% **không ràng buộc** (traffic đã dưới mức đó); cap 5% tiết kiệm $9/tháng + **11,934 Wh/ngày**; cap 2% → $17/tháng + 22,543 Wh/ngày. |
| 5 | Carbon-aware scheduling | `missions/m6_carbon_scheduling.py` | Chuyển 5 job interruptible sang `europe-north1`: **−532 kgCO2e/tháng (−92%)** và **−$45.63/tháng** tiền điện. 3 job serving phải ở lại vì +110ms là độ trễ người dùng thấy được. |

**Insight quan trọng nhất từ extensions:** Extension 1 cho thấy một "cải tiến" đúng đắn có thể làm con số tiết kiệm *giảm xuống* — vì nó xoá đi phần tiết kiệm không có thật. Với FinOps, số liệu đúng quan trọng hơn số liệu đẹp; nếu đem con số v1 đi cam kết với CFO thì mỗi tháng sẽ hụt $324 và không ai giải thích được.

## 5. Tính bền vững — carbon gắn với tiền thật

| Vùng | $/kWh | gCO2/kWh | Độ trễ thêm | Kết luận |
|---|---|---|---|---|
| europe-north1 | 0.090 | 30 | +110ms | Sạch nhất — đặt training interruptible |
| us-east-wa | 0.055 | 90 | +55ms | **Rẻ nhất VÀ gần sạch nhất — lựa chọn cân bằng** |
| us-west-2 | 0.070 | 120 | +70ms | Phương án dự phòng ở Mỹ |
| us-east-1 | 0.120 | 380 | +5ms | Nơi đang chạy hôm nay |
| europe-central2 | 0.180 | 660 | +120ms | Tránh — vừa bẩn nhất vừa đắt nhất |

Carbon và chi phí **không mâu thuẫn** trong bảng này: `us-east-wa` vừa rẻ hơn us-east-1 54% tiền điện, vừa sạch hơn 4.2×. Ràng buộc thật sự là **độ trễ**, không phải tiền. Vì vậy chính sách hợp lý là tách đôi: job *chuyển được* (interruptible, không ai chờ) đi theo lưới điện sạch/rẻ; job *serving* ở lại cạnh người dùng. Với "vùng tối ưu" thì không có câu trả lời duy nhất — nếu ưu tiên phát thải chọn `europe-north1`, nếu ưu tiên hoá đơn chọn `us-east-wa`, nếu ưu tiên trải nghiệm người dùng thì buộc phải ở lại `us-east-1` và bù bằng cách giảm Wh/query (chính là Extension 4).

## 6. Nếu tôi là FinOps lead của NimbusAI — 3 hành động đầu tiên

1. **Bật cascade routing làm mặc định trong tuần đầu ($829/tháng, vài ngày công).** Đây là đòn bẩy đơn lẻ lớn nhất về $/1M-token, không cần đàm phán với nhà cung cấp, và bật/tắt được theo từng route nên rủi ro thấp — sai thì rollback trong vài phút. Kèm theo điều kiện thoát: nếu tỷ lệ escalate lên model lớn vượt 25% thì xem lại ngưỡng.
2. **Thay chỉ số điều hành từ GPU-Util sang MFU/MBU, và bật auto-stop GPU idle ($600/tháng, vài giờ công).** Auto-stop là thứ rẻ nhất trong cả báo cáo, nhưng giá trị lớn hơn nằm ở việc đổi thước đo: chừng nào dashboard còn báo `gpu-h100-4` là 98% "đã dùng", chúng ta còn tiếp tục mua sai SKU. Đặt alert `util>90% AND MFU<0.30` ngay từ ngày đầu.
3. **Chuyển 5 job interruptible sang spot + checkpoint, và chỉ cam kết reserved cho 3 job serving 24/7 ($9,788/tháng, 1–2 tuần).** Giá trị lớn nhất nhưng để sau vì cần hạ tầng checkpoint hoạt động trước — spot không có checkpoint là đánh cược, không phải tối ưu. Trước khi ký reserved, dùng `recommend_tier_v2()` (tính theo 720h cam kết) chứ không dùng công thức theo giờ dùng thật.

Song song (chi phí gần bằng 0): bật **chargeback** — tag coverage đang là **92%**, đã qua cổng 80%. Chi phí inference hiện chia theo team là assistant $2.59/ngày, search $2.49, eval $1.79, rag $1.60. Lý do cần ≥80% coverage trước khi thu tiền: dưới ngưỡng đó, phần "(untagged)" đủ lớn để một team có thể phản bác hoá đơn của mình — và một lần tranh cãi thua là chương trình chargeback chết. Dữ liệu đã xuất sẵn theo chuẩn **FOCUS** (`outputs/focus_export.csv`) để khi thêm nhà cung cấp thứ hai thì không phải viết lại pipeline phân bổ.

---

## Phụ lục — kiểm tra tự động

```
python verify.py   ->  11/11 checks passed
pytest -q          ->  33 passed  (15 test gốc + 18 test tự viết cho extensions)
```

Không sửa file test gốc; các test mới nằm trong `tests/test_extensions.py`, bao gồm cả một test kiểm tra **tính nhất quán giữa `outputs/report.md` và output của các mission** (`test_report_numbers_match_mission_outputs`).
