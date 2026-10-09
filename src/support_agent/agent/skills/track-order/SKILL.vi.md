---
name: track-order
description: Tìm vị trí đơn hàng và thời gian dự kiến giao tới
lang: vi
---

# Theo dõi đơn hàng

1. Lấy mã đơn từ khách. Nếu khách chưa có mã, gọi `list_orders` (dùng `status` để thu hẹp, ví dụ `shipping`) rồi để khách chọn.
2. Gọi đồng thời `get_order` và `get_shipment_status` cho mã đó.
3. Trả lời theo thứ tự: trạng thái đơn, đơn vị vận chuyển và mã vận đơn, trạng thái vận chuyển, ngày dự kiến giao nếu có.

Nếu `shipments` liệt kê nhiều kiện, đơn đã được tách: nêu lần lượt từng kiện (đơn vị vận chuyển, mã vận đơn, trạng thái) và nói đã giao đủ chưa (`all_delivered`).

Nếu chưa có vận đơn:
- `processing` (đang xử lý): cửa hàng chưa bàn giao cho đơn vị vận chuyển. Nói rõ điều này, và dùng `search_policy` nếu khách hỏi sẽ được báo bằng cách nào khi đơn được giao đi.
- `pending_payment` (chờ thanh toán): đơn đang chờ thanh toán và sẽ không được giao cho tới khi thanh toán.
- `on_hold` (tạm giữ): cửa hàng đang tạm dừng đơn; không đoán lý do, gợi ý khách liên hệ hỗ trợ.
- `cancelled` (đã hủy): đơn đã bị hủy, sẽ không có hàng được giao.
- `returned` / `refunded` (đã trả hàng / đã hoàn tiền): hàng đã được trả lại hoặc tiền đã hoàn; sẽ không giao thêm.

Nếu trạng thái là `partially_shipped` (giao một phần): một phần đơn đã được giao, nói rõ phần nào theo dữ liệu và phần còn lại sẽ giao sau.

Nếu không tìm thấy đơn, nói rằng bạn không tìm thấy đơn đó trong tài khoản của khách và nhờ khách kiểm tra lại mã. Không đoán khách đang nói đến đơn nào.

Nếu giao chậm: chính sách quy định khi nào có thể báo kiện hàng trễ cho bộ phận hỗ trợ. Dùng `search_policy` để trích đúng quy định thay vì nhớ lại.
