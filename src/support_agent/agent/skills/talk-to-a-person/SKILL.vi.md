---
name: talk-to-a-person
description: Chuyển khách cho nhân viên của cửa hàng
lang: vi
requires: request:handoff
---

# Chuyển khách cho nhân viên

Dùng khi khách muốn gặp người thật, đang bức xúc và cần người xử lý, hoặc bạn không giải quyết được điều khách cần (công cụ báo không hỗ trợ, quy tắc không thể quyết định, chính sách không đề cập). Không dùng khi bạn có thể tự trả lời hoặc tự làm được.

1. Nói trong một câu điều bạn không làm được cho khách ở đây. Không xin lỗi dài dòng.
2. Nếu chưa rõ, hỏi khách cần gì bằng lời của chính họ, và đơn hàng liên quan nếu có. Lý do phải là của khách: không tự viết thay họ.
3. Gọi `propose_draft` với `draft_type` là "handoff", `reason_text` (khách cần gì, bằng lời của khách), `order_id` nếu liên quan, và `contact` chỉ khi khách tự cho email hoặc số điện thoại để liên hệ lại. Không hỏi thông tin liên hệ mà khách chưa tự đưa ra.
4. Khách sẽ được hỏi xác nhận. Chỉ khi `propose_draft` báo đã tạo, mới nói mã yêu cầu và rằng nhân viên sẽ liên hệ lại. Không hứa thời gian hay kết quả.

Nếu thông tin liên hệ của cửa hàng có trong hướng dẫn của bạn, hãy đưa kèm giờ làm việc để khách có thể tự liên hệ trực tiếp. Không đưa bất kỳ thông tin liên hệ nào khác.
