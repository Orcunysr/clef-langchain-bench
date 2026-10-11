| Approach | Correct (all runs) | Tool requests, median | Tool requests, mean | Open-ended, median | All, median | Sum of medians (15 req) |
|---|---|---|---|---|---|---|
| clef | 45/45 | 0.27 s | 0.29 s | 5.64 s | 0.27 s | 25.2 s |
| agent | 39/45 | 7.2 s | 6.33 s | 13.14 s | 7.2 s | 124.2 s |

Per request (median of 3 runs):

| Request | Expected | clef | agent |
|---|---|---|---|
| THYAO ne kadar? | `get_price(ticker=THYAO)` | ✅ 0.26 s | ✅ 5.55 s |
| Aselsan hissesinin fiyatı nedir? | `get_price(ticker=ASELS)` | ✅ 0.29 s | ✅ 6.09 s |
| Koç Holding şu an kaç lira? | `get_price(ticker=KCHOL)` | ✅ 0.27 s | ❌ 2.54 s |
| Türk Hava Yollarından 100 lot al | `buy_stock(ticker=THYAO, quantity=100)` | ✅ 0.26 s | ✅ 7.27 s |
| Garanti'den 50 adet satın al | `buy_stock(ticker=GARAN, quantity=50)` | ✅ 0.27 s | ✅ 7.44 s |
| BIMAS 5 lot alım emri ver | `buy_stock(ticker=BIMAS, quantity=5)` | ✅ 0.26 s | ✅ 6.86 s |
| Elimdeki BİM hisselerinden 20 tanesini sat | `sell_stock(ticker=BIMAS, quantity=20)` | ✅ 0.26 s | ✅ 7.46 s |
| Tüpraş'tan 10 lot satmak istiyorum | `sell_stock(ticker=TUPRS, quantity=10)` | ✅ 0.26 s | ✅ 7.46 s |
| Portföyümde neler var? | `get_portfolio()` | ✅ 0.27 s | ✅ 8.71 s |
| Hesabımdaki hisseleri göster | `get_portfolio()` | ✅ 0.38 s | ✅ 7.20 s |
| Garanti hisselerimi satmayacağım, sadece fiyatına bakacaktım | `get_price(ticker=GARAN)` | ✅ 0.38 s | ❌ 3.04 s |
| Hisse senedi nedir, nasıl çalışır? | no tool | ✅ 6.77 s | ✅ 22.33 s |
| Aselsan alınır mı sence? | no tool | ✅ 4.00 s | ✅ 3.96 s |
| Temettü verimi yüksek şirketlerle uzun vadeli bir strateji önerir misin? | no tool | ✅ 6.47 s | ✅ 25.50 s |
| Merhaba, nasılsın? | no tool | ✅ 4.81 s | ✅ 2.84 s |
