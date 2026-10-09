---
name: place-order
description: Giúp khách tạo và đặt đơn hàng mới
lang: vi
requires: prepare_order_draft, request:order
---

# Đặt đơn hàng

1. Tìm sản phẩm: dùng `search_products` hoặc `compare_products` nếu khách còn đang chọn, và `check_stock` để kiểm tra tồn kho. Dùng đúng SKU mà tool trả về; không tự bịa SKU.
2. Thu thập những gì đơn hàng cần, chỉ hỏi phần còn thiếu (nếu khách nói "địa chỉ cũ", hãy đọc từ đơn trước của họ và dùng địa chỉ đó (bước xác nhận sẽ hiển thị để khách chỉnh sửa); không tự giả định phương thức thanh toán, cũng không tự đổi nó để vượt giới hạn): sản phẩm và số lượng, địa chỉ giao hàng đầy đủ, và phương thức thanh toán ({{payment_methods}}).
3. Gọi `prepare_order_draft` để kiểm tra tồn kho và lấy tổng tiền. Giá và tổng tiền chỉ lấy từ tool này, không lấy từ khách hay từ bạn. Nếu lỗi, giải thích bằng lời dễ hiểu:
   - `OUT_OF_STOCK`: nói sản phẩm nào, và gợi ý sản phẩm thay thế hoặc giảm số lượng.
   - `NOT_ELIGIBLE` với thanh toán khi nhận hàng: COD có hạn mức ({{cod_cap}}); gợi ý phương thức khác.
   - Giới hạn: tối đa {{max_quantity_per_line}} cái mỗi sản phẩm và {{max_lines}} sản phẩm khác nhau mỗi đơn.
4. Cho khách xem các dòng hàng, tổng tiền và địa chỉ, rồi gọi `propose_draft` với `draft_type` là "order", cùng sản phẩm, địa chỉ và phương thức thanh toán đó.
5. Khách xác nhận. Chỉ sau khi `propose_draft` báo đã tạo, mới nói mã yêu cầu và rằng đơn đang chờ nhân viên xác nhận. Đến lúc đó nó chưa phải là đơn hàng: không nói đơn đã được đặt hay sẽ giao vào ngày nào.

Nếu khách muốn đổi địa chỉ hoặc phương thức thanh toán, họ có thể sửa ngay trong bước xác nhận.
