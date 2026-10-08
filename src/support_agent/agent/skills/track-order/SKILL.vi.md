---
name: track-order
description: Tìm vị trí đơn hàng và thời gian dự kiến giao tới
lang: vi
---

# Theo dõi đơn hàng

1. Lấy mã đơn từ khách. Nếu khách chưa có mã, gọi `list_orders` (dùng `status` để thu hẹp, ví dụ `shipping`) rồi để khách chọn.
2. Gọi đồng thời `get_order` và `get_shipment_status` cho mã đó.
3. Trả lời theo thứ tự: trạng thái đơn, đơn vị vận chuyển và mã vận đơn, trạng thái vận chuyển, ngày dự kiến giao nếu có.

Nếu chưa có vận đơn:
- `processing` (đang xử lý): cửa hàng chưa bàn giao cho đơn vị vận chuyển. Nói rõ điều này, và mã vận đơn sẽ được gửi qua email và SMS khi đơn được giao đi.
- `cancelled` (đã hủy): đơn đã bị hủy, sẽ không có hàng được giao.

Nếu không tìm thấy đơn, nói rằng bạn không tìm thấy đơn đó trong tài khoản của khách và nhờ khách kiểm tra lại mã. Không đoán khách đang nói đến đơn nào.

Nếu giao chậm: chính sách cho phép liên hệ hỗ trợ khi kiện hàng trễ 3 ngày làm việc so với ngày dự kiến. Dùng `search_policy` để trích đúng quy định thay vì nhớ lại.
