---
name: return-item
description: Kiểm tra đơn có đổi trả được không và tạo yêu cầu đổi trả
lang: vi
requires: request:return
---

# Đổi trả sản phẩm

1. Lấy mã đơn, và sản phẩm nếu đơn có nhiều món.
2. Gọi `check_return_eligibility`. Kết luận của nó là quyết định cuối: không tự đánh giá điều kiện.
3. Giải thích kết luận:
   - Đủ điều kiện: nói còn bao nhiêu ngày, hạn chót, và số tiền được hoàn.
   - Không đủ điều kiện: nêu lý do bằng lời dễ hiểu. Nếu kết quả có `reason_windows` với `still_open`, thì kết luận còn phụ thuộc vào lý do trả hàng (ví dụ sản phẩm lỗi có thể được thời hạn dài hơn): nói rõ điều đó và hỏi lý do, rồi gọi lại `check_return_eligibility` với `reason`. Ý nghĩa mã lý do: `WINDOW_EXPIRED` đã hết thời hạn đổi trả, `STATUS_NOT_ALLOWED` đơn chưa được giao, `CATEGORY_EXCLUDED` loại sản phẩm này không đổi trả được (ví dụ thẻ quà tặng), `ALREADY_REQUESTED` đã có yêu cầu đang mở cho sản phẩm này, `ITEM_NOT_FOUND` sản phẩm không có trong đơn.
4. Nếu đủ điều kiện và khách muốn tiếp tục, hỏi lý do (lỗi, giao sai hàng, không đúng mô tả, hỏng khi vận chuyển, đổi ý, khác) và mô tả ngắn. Sau đó gọi `propose_draft` với `draft_type` là "return". Nếu khách muốn hoàn tiền thì dùng skill `request-refund`.
5. Khách sẽ được hỏi xác nhận. Không nói yêu cầu đã được gửi cho đến khi `propose_draft` báo đã tạo. Sau đó nói mã yêu cầu và rằng nhân viên sẽ xem xét (dùng `search_policy` nếu khách hỏi mất bao lâu).

Dùng `search_policy` để trích đúng nội dung trước khi hứa điều gì về phí vận chuyển hay đổi sản phẩm.
