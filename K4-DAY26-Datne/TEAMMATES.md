# TEAMMATES.md

## Thông tin đội thi đấu
- **Tên đội:** Datne
- **Khóa học:** K4 (AI20K - Level 3)
- **Bài tập:** Day 26 — Colosseum Agent Arena

## Danh sách thành viên
| STT | Họ và Tên | MSSV | Email | Vai trò | Tỷ lệ đóng góp |
|:---:|:---|:---:|:---|:---|:---:|
| 1 | Nguyễn Tiến Đạt | 2A202602970 | dn841746@gmail.com | Trưởng nhóm / Toàn bộ bài nộp | 100% |

## Thành phần bài nộp
- `Datne.bundle`: Gói nộp bài niêm phong (sealed bundle) tạo bởi `kit.submit`, bao gồm:
  - `agent/`: Hệ thống Gateway phòng thủ, guardrails, telemetry, prompt, strategy.
  - `deck/`: Bộ bài đấu 14 lá (10 attack + 4 blank) hợp lệ và lineup chiến thuật đã vượt qua `validate_deck.py`.
  - `eval/`: Công tố viên (`eval/prosecute.py`) với đầy đủ 9 detector tất định đạt 100% precision và recall.
  - `MANIFEST.json`: Chữ ký số SHA-256 niêm phong toàn bộ các file `kit/` nguyên bản.
