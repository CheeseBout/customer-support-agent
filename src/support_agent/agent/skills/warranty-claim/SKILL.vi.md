---
name: warranty-claim
description: Kiểm tra bảo hành cho sản phẩm và tạo yêu cầu bảo hành
lang: vi
requires: request:warranty
---

# Yêu cầu bảo hành

1. Lấy mã đơn, sản phẩm (SKU) và tình trạng hỏng hóc. Nếu lỗi nằm ở một bộ phận (pin, màn hình, dây sạc), yêu cầu bảo hành là cho sản phẩm chứa bộ phận đó: lấy SKU từ `get_order`, đừng tìm bộ phận ấy trong danh mục. Nếu khách chưa nói bị lỗi gì, hãy hỏi trước khi đề xuất.
2. Gọi `check_warranty_eligibility` với mã đơn và SKU. Kết luận của nó là quyết định cuối.
3. Giải thích: thời hạn bảo hành của loại sản phẩm này (mặc định 12 tháng, phụ kiện 6 tháng, đồ gia dụng 24 tháng), bảo hành hết hạn khi nào và còn bao lâu. Mã lý do: `WARRANTY_EXPIRED` đã hết bảo hành, `NOT_DELIVERED` bảo hành tính từ ngày giao hàng, `ITEM_NOT_FOUND` sản phẩm không có trong đơn.
4. Nếu còn bảo hành và khách muốn tiếp tục, gọi `propose_draft` với `draft_type` là "warranty", mã đơn, SKU và `issue_description` theo lời khách.
5. Khách xác nhận. Chỉ sau khi `propose_draft` báo đã tạo, mới đưa mã yêu cầu và nói nhân viên sẽ hướng dẫn cách gửi hoặc mang sản phẩm đến trung tâm bảo hành.

Những trường hợp không được bảo hành (hư hỏng vật lý, vào nước, bị sửa bởi nơi khác, hao mòn thông thường) nằm trong chính sách: dùng `search_policy` và đọc lại chính xác nếu khách mô tả hư hỏng như vậy. Dùng `search_policy` để biết thời gian sửa chữa; không hứa đổi sản phẩm mới vì điều đó được quyết định sau khi kiểm tra.
