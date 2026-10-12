---
name: exchange-item
description: Kiểm tra sản phẩm có đổi được không (đổi size, màu hoặc đổi sang sản phẩm mới) và tạo yêu cầu đổi hàng
lang: vi
requires: request:exchange
---

# Đổi sản phẩm

Đổi hàng là thay sản phẩm khách đã nhận bằng một sản phẩm khác: size hoặc màu khác của cùng một mẫu, hoặc một sản phẩm mới thay cho sản phẩm bị lỗi. Đây KHÔNG phải là trả hàng. Đổi hàng có thời hạn và quy tắc riêng, nên tuyệt đối không dùng `check_return_eligibility` hay thời hạn trả hàng để đánh giá.

1. Lấy mã đơn và sản phẩm cần đổi (lấy SKU từ `get_order`). Nếu khách đã nói lý do (không vừa, bị lỗi, giao sai hàng) thì giữ lại và truyền vào `reason` ngay từ lần gọi đầu. Size hoặc màu không phù hợp là `changed_mind`.
2. Gọi `check_exchange_eligibility` với mã đơn, `sku` của sản phẩm và `reason` nếu đã biết. Kết luận của nó là quyết định cuối: không tự đánh giá điều kiện.
3. Giải thích kết luận:
   - Đủ điều kiện: nói hạn chót và còn bao nhiêu ngày.
   - Không đủ điều kiện: nêu lý do bằng lời dễ hiểu. Ý nghĩa mã lý do: `WINDOW_EXPIRED` đã hết thời hạn đổi hàng, `STATUS_NOT_ALLOWED` đơn chưa được giao, `CATEGORY_EXCLUDED` loại sản phẩm này không đổi được, `EXCHANGE_LIMIT_REACHED` đơn đã dùng hết số lần đổi cho phép, `ALREADY_REQUESTED` sản phẩm đã nằm trong một yêu cầu đang mở, `ITEM_NOT_FOUND` sản phẩm không có trong đơn. Nếu kết quả có `reason_windows` với `still_open`, thì kết luận còn phụ thuộc vào lý do đổi (ví dụ sản phẩm lỗi có thể được thời hạn dài hơn): nói rõ điều đó, hỏi lý do rồi gọi lại với `reason`.
4. Nếu đủ điều kiện, hỏi khách muốn đổi sang gì nếu họ chưa nói (size hoặc màu). Dùng `check_stock` hoặc `search_products` theo tên sản phẩm để tìm SKU của biến thể đó và xem còn hàng không, rồi gọi lại `check_exchange_eligibility` với `new_sku` để xác nhận có thể đổi sang. Nếu hết hàng hoặc không cùng sản phẩm, nói rõ và gợi ý size hoặc màu khác.
5. Gọi `propose_draft` với `draft_type` là "exchange", mã đơn, `sku`, `new_sku`, `reason_code` và `reason_text` ngắn bằng lời của khách.
6. Khách sẽ được hỏi xác nhận. Chỉ khi `propose_draft` báo đã tạo, mới nói mã yêu cầu và rằng nhân viên sẽ xem xét. Không hứa chắc sẽ được duyệt. Dùng `search_policy` cho phí đổi, ai chịu phí vận chuyển và cách gửi sản phẩm mới; không tự nêu những điều đó theo trí nhớ.
