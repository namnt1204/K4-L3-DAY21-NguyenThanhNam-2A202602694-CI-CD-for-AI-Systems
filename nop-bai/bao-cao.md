# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Nguyễn Thành Nam |
| MSSV | 2A202602694 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/namnt1204/K4-L3-DAY21-NguyenThanhNam-2A202602694-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ tham số này đạt điểm F1 cao nhất (0.7149), vượt qua ngưỡng chất lượng 0.65 của hệ thống. Lần chạy có accuracy cao nhất là lần 1 (0.8780) nhưng F1 (0.7109) lại thấp hơn lần 3. Điều này phản ánh rõ ràng việc accuracy không tối ưu cho lớp thiểu số. Giữa `n_estimators` và `learning_rate` có sự đánh đổi: khi giảm cả tốc độ học xuống 0.05 và số cây xuống 50 (lần 2), mô hình bị underfitting nghiêm trọng khiến F1 giảm mạnh xuống 0.6051 và bị chặn bởi Quality Gate. Ngược lại, tăng số cây lên 200 kết hợp độ sâu cây 5 giúp mô hình nắm bắt đầy đủ các tương tác đặc trưng phức tạp.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult Income có sự mất cân bằng lớp rõ rệt với tỷ lệ lớp dương (thu nhập > 50K) chỉ chiếm khoảng 24% đến 25%. Một mô hình ngây thơ (naive baseline) luôn dự đoán nhãn "thu nhập thấp" cho toàn bộ mẫu vẫn có thể dễ dàng đạt Accuracy 75%, nhưng hoàn toàn vô giá trị trong thực tế vì Recall bằng 0 và không phát hiện được bất kỳ người có thu nhập cao nào.

Chỉ số F1-score của lớp dương là trung bình điều hòa giữa Precision và Recall cho nhãn mục tiêu, phản ánh chính xác năng lực phát hiện trường hợp thu nhập cao mà không bị đánh lừa bởi số lượng áp đảo của lớp âm. Bài toán không sử dụng `average="macro"` hay `weighted` vì các trung bình này sẽ bị pha loãng bởi độ chính xác cao của lớp thu nhập thấp, làm suy giảm mục tiêu cốt lõi của bài toán dự báo.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| DVC pull báo lỗi `HeadObject 400 Bad Request` trên GitHub Actions. | Ghi biến môi trường vào `$GITHUB_ENV` chưa có hiệu lực trong cùng step, khiến lệnh DVC nhận chuỗi rỗng. | Cấu hình trực tiếp biến xác thực qua Python subprocess và bổ sung biến môi trường rõ ràng cho step. |
| Lỗi unpickle `CyHalfBinomialLoss` khi service khởi động trên EC2. | Máy ảo Ubuntu chạy Python 3.14 với phiên bản scikit-learn mới hơn môi trường huấn luyện. | Bổ sung adapter tương thích cho Cython loss function trong `src/serve.py` trước khi gọi `joblib.load`. |
| Health check bị timeout khi khởi động lại API trên máy ảo. | Thời gian tải artifact từ S3 và nạp cây quyết định vượt quá ngưỡng `sleep 5` cố định. | Thay thế lệnh chờ cố định bằng vòng lặp thăm dò (polling retry) kiểm tra endpoint `/healthz` tối đa 30 giây. |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.8820 |

**Nhận xét:** Khi bổ sung thêm 22.361 mẫu dữ liệu ở Bước 3 (tổng 44.722 mẫu), F1-score tăng từ 0.7149 lên 0.7354 và accuracy tăng từ 0.8740 lên 0.8820 trên cùng tập holdout được giữ cố định. Do dữ liệu bổ sung có cùng phân phối, lượng mẫu tăng gấp đôi giúp mô hình học ranh giới quyết định chính xác hơn và chứng minh toàn bộ quy trình CI/CD kích hoạt hoàn toàn tự động từ commit tệp pointer `.dvc`.
