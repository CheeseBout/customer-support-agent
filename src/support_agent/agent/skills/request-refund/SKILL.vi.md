---
name: request-refund
description: Yêu cầu hoàn tiền cho sản phẩm trả lại hoặc bị lỗi
lang: vi
---

# Yêu cầu hoàn tiền

1. Lấy mã đơn, sản phẩm và lý do. Lý do phải là của khách: nếu khách chỉ nói muốn hoàn tiền, hãy hỏi vì sao và không gọi `propose_draft` với một lý do chung chung. Kiểm tra điều kiện bằng `check_return_eligibility` đúng như skill `return-item`; quy tắc đổi trả và hoàn tiền là một.
2. Không tự nêu số tiền hoàn. Số tiền do bộ quy tắc tính (`refundable_amount`) và được hiển thị cho khách khi xác nhận. Bạn chỉ được nhắc lại số đó sau khi tool trả về.
3. Gọi `propose_draft` với `draft_type` là "refund", mã đơn, mã lý do và lý do ngắn. Chỉ truyền `sku` hoặc `items` nếu khách chỉ muốn hoàn một phần đơn.
4. Khách xác nhận. Chỉ khi `propose_draft` báo draft đã được tạo, mới báo khách: mã yêu cầu, yêu cầu đang chờ nhân viên xem xét, và tiền được hoàn trong 5 đến 7 ngày làm việc sau khi được duyệt và nhận lại hàng (dùng `search_policy` để xác nhận nội dung hiện hành).

Yêu cầu hoàn tiền trên 500.000đ được nhân viên cấp cao xem xét ưu tiên: có thể nhắc điều này khi số tiền lớn, nhưng tuyệt đối không hứa chắc sẽ được duyệt. Yêu cầu hoàn tiền chưa phải là đã hoàn tiền cho đến khi nhân viên duyệt.

Nếu khách từ chối xác nhận, hãy chấp nhận và không thúc ép. Nếu khách muốn đổi lý do, họ có thể sửa ngay ở bước xác nhận.
