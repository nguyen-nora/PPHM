# Phương pháp học máy - 2024
## Resnet
- Tải model train [object_detection.keras](https://drive.google.com/file/d/19OVQaVZdqQz0rBdo-O8uyetYHlRnSiU6/view?usp=drive_link)
- Thêm vào thư mục Resnet - LPD
- Đổi tên file hình ảnh muốn chạy ở phần img_path
## VGG16
- Đổi tên file hình ảnh muốn chạy ở phần img_path
## App + Yolov8
*Lưu ý: Thay đổi server, database, username, password database và driver ODBC trong license_plate_DB.py. Nếu bạn không có tài khoản hoặc không muốn đăng nhập, phần mềm sẽ mất khoảng 15 giây để khởi động vì không thể kết nối với cơ sở dữ liệu.*

**Hiện tại, phần mềm đang gặp nhiều lỗi và vấn đề về hiệu suất. Chúng tôi mong nhận được sự đóng góp và lời khuyên từ mọi người.**

*Nếu bạn nhấn nút quick_view, một cửa sổ mới sẽ xuất hiện để người khác dễ dàng xem, tuy nhiên nó đang gây ra lỗi khiến một trong hai camera đầu tiên trong cửa sổ chính bị đóng băng hình ảnh. Có một lỗi rò rỉ RAM với tần suất tăng 0,1% trong 1 phút, tôi nghĩ vấn đề này là do sử dụng cv2. Khi đóng gói với auto-py-to-exe, sử dụng chức năng quẹt thẻ lần đầu tiên sẽ khiến ứng dụng mở lại.*

