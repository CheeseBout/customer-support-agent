---
name: return-item
description: Kiểm tra đơn có đổi trả được không và tạo yêu cầu đổi trả
lang: vi
---

# Đổi trả sản phẩm

1. Lấy mã đơn, và sản phẩm nếu đơn có nhiều món.
2. Gọi `check_return_eligibility`. Kết luận của nó là quyết định cuối: không tự đánh giá điều kiện.
3. Giải thích kết luận:
   - Đủ điều kiện: nói còn bao nhiêu ngày, hạn chót, và số tiền được hoàn.
   - Không đủ điều kiện: nêu lý do bằng lời dễ hiểu. Ý nghĩa mã lý do: `WINDOW_EXPIRED` đã hết thời hạn đổi trả, `STATUS_NOT_ALLOWED` đơn chưa được giao, `CATEGORY_EXCLUDED` loại sản phẩm này không đổi trả được (ví dụ thẻ quà tặng), `ALREADY_REQUESTED` đã có yêu cầu đang mở cho sản phẩm này, `ITEM_NOT_FOUND` sản phẩm không có trong đơn.
4. Nếu đủ điều kiện và khách muốn tiếp tục, hỏi lý do (lỗi, giao sai hàng, không đúng mô tả, hỏng khi vận chuyển, đổi ý, khác) và mô tả ngắn. Sau đó gọi `propose_draft` với `draft_type` là "return". Nếu khách muốn hoàn tiền thì dùng skill `request-refund`.
5. Khách sẽ được hỏi xác nhận. Không nói yêu cầu đã được gửi cho đến khi `propose_draft` báo đã tạo. Sau đó nói mã yêu cầu và rằng nhân viên sẽ xem xét, thường trong 1 ngày làm việc.

Phí vận chuyển trả hàng chỉ được miễn với sản phẩm lỗi hoặc hỏng. Dùng `search_policy` để trích đúng nội dung trước khi hứa điều gì về phí vận chuyển hay đổi sản phẩm.
