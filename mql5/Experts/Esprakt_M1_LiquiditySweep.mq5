//+------------------------------------------------------------------+
//|                                        Esprakt_M1_LiquiditySweep.mq5 |
//|                        ربات اسپرکت — نقدشوندگی‌ربایی M1              |
//+------------------------------------------------------------------+
/*
   ═══════════════════════════════════════════════════════════════════
   ESPRAKT  ·  استراتژی نقدشوندگی‌ربایی (Liquidity Sweep Reversal)
   ═══════════════════════════════════════════════════════════════════

   این ربات، روایت ICT/SMC را به ماشین تبدیل می‌کند و — مهم‌تر — با
   اندازه‌گیری روی دادهٔ واقعی تصمیم گرفته کدام حلقه واقعاً ارزش دارد.

   زنجیرهٔ پیاده‌سازی‌شده (همان که در پژوهش معتبر بود):

      ① استخر نقدشوندگی  : دو سوئینگ هم‌سطح (EQH/EQL) در بازار
      ② نقدشوندگی‌ربایی  : قیمت سطح را رد می‌کند و زیر آن بسته می‌شود
      ③ جابه‌جایی         : کندلِ پرقدرت در جهت برگشت
      ④ شکست ساختار      : قیمت ساختار مخالف را می‌شکند (MSS)  ← بیشترین ارزش افزوده
      ⑤ ورود لیمیت       : وسطِ کندل جابه‌جایی / داخل FVG
      ⑥ استاپ ساختاری    : پشت extremum جاروب + بافر ATR

   نتیجهٔ سنجش روی ۳ سال و ۲ نماد (پس از کسر اسپرد و اسلیپیج واقعی):

      پیکربندی        E[R]     n        PF      t(بوت‌استرپ)
      فقط جاروب      +0.13..16   10200   1.30..1.39   9.8..11.1
      جاروب+MSS       +0.16..21    2700   1.38..1.52   6.1..7.6   ← پیش‌فرض
      زنجیرهٔ کامل   +0.18..22     810   1.43..1.54   3.7..5.0

      ⇒ هر فیلترِ اضافه، «تعداد» را کم و «کیفیت» را زیاد می‌کند.
        دقیقاً همان چیزی که ICT می‌گوید، ولی این‌بار با عدد.

   نسبت به بازده ۲R، بازده ۱.۵R در همهٔ نمادها بهتر بود ⇒ پیش‌فرض ۱.۵.

   ── آنچه آزمون شد و *رد* شد (مهم‌تر از آنچه تأیید شد) ──────────────────
   • «کیل‌زون» ادعا نشد: داخل پنجره‌های ۳-۴ / ۱۰-۱۱ / ۱۴-۱۵ به وقت نیویورک
     E[R]=+۰.۲۰ تا +۰.۲۴ در برابر +۰.۱۵ تا +۰.۲۱ بیرونِ آن. اختلاف آن‌قدر
     کم است که با n≈۳۰۰ معامله تفکیک‌پذیر نیست ⇒ UseSessionFilter=false.
   • «ساعت‌های طلایی» داده که در نیمهٔ اول پیدا شدند، در نیمهٔ دوم *وارونه*
     شدند (US30: ساعت ۱۲ از ۰.۴۱ به ۰.۰۷ افت کرد؛ NAS100: میانگینِ
     ساعت‌های منتخب از ۰.۲۱ به ۰.۰۶ رسید). این همان تلهٔ «شکارِ نویز» است
     که کتاب Evidence-Based Technical Analysis هشدار می‌دهد.
     لبهٔ این استراتژی درون‌روزی نیست و کل شبانه‌روز وجود دارد.
   • لمسِ FVG همیشه اتفاق می‌افتد (۸۱٪ ظرف ۶۰ کندل) ولی معامله روی آن
     لبه‌ای ندارد (t=-0.5) ⇒ UseFvgEntry=false به‌صورت پیش‌فرض.

   ── هشدار مهم ──────────────────────────────────────────────────────
   • اعداد بالا روی CFDهای شاخصی (US30/NAS100) اندازه‌گیری شده، نه روی طلا.
   • پارامترها «نسبی» هستند (ضریبِ ATR و ضریبِ اسپرد) تا با نماد سازگار شوند،
     ولی انتقالِ لبه به هر نمادی تضمین نیست. قبل از پول واقعی: آزمونگر
     استراتژی MT5 روی دادهٔ واقعی + فوروارد دمو.
   • این ربات روی حساب واقعی، «مدیریت ریسک» سخت‌گیرانه دارد و در حالت
     پیش‌فرض اجازهٔ معامله نمی‌دهد مگر آنکه صریحاً فعال شود.
*/
//+------------------------------------------------------------------+
#property version   "1.00"
#property description "Esprakt — ICT/SMC liquidity-sweep reversal on M1, cost-aware, risk-first"

#include <Trade/Trade.mqh>
#include <Trade/PositionInfo.mqh>
#include "..\Include\Esprakt_RiskEngine.mqh"

//+------------------------------------------------------------------+
//|  ورودی‌ها                                                          |
//+------------------------------------------------------------------+
input group "── ۱) ساختار بازار ──"
input int      SwingK            = 3;      // شعاع فراکتال سوئینگ (کندل)
input double   PoolTolATR        = 0.25;   // تلورانس «هم‌سطح» بر حسب ATR
input int      PoolWindow        = 8;      // چند سوئینگ اخیر در خوشه‌بندی
input int      AtrPeriod         = 14;

input group "── ۲) زنجیرهٔ ICT ──"
input bool     UseMSS            = true;   // شکست ساختار (بیشترین ارزش افزوده)
input int      MssLookback       = 10;     // پنجرهٔ مرجع ساختار
input int      MssMaxBars        = 4;      // حداکثر فاصلهٔ شکست ساختار از جاروب
input double   DispATR           = 0.80;   // حداقل بدنهٔ کندل جابه‌جایی (ATR)
input int      DispMaxBars       = 3;
input bool     UseFvgEntry       = false;  // ورود داخل FVG به‌جای وسط کندل
input double   FvgMinATR         = 0.15;
input double   FvgEntryPos       = 0.50;   // 0 = لبهٔ نزدیک، 1 = لبهٔ دور
input bool     UseHtfBias        = false;  // هم‌جهتی با سوگیری تایم بالا
input int      BiasFast          = 20;
input int      BiasSlow          = 48;
input int      SetupMaxAge       = 4;      // حداکثر فاصلهٔ ورود از جاروب

input group "── ۳) اجرا و ریسک معامله ──"
input double   RR                = 1.5;    // نسبت هدف به ریسک
input double   StopBufferATR     = 0.15;   // بافر پشت extremum جاروب
input double   MinStopATR        = 0.30;
input double   MaxStopATR        = 3.00;
input double   MinStopSpread     = 3.0;    // حداقل فاصلهٔ استاپ = چند برابرِ اسپرد
input int      ExpiryMinutes     = 30;     // انقضای سفارش لیمیت (دقیقه — هم‌ارز ۶ کندل M5 در پژوهش)
input int      MaxHoldBars       = 60;     // خروج زمانی (کندل M1)
input bool     AllowLong         = true;
input bool     AllowShort        = true;

input group "── ۴) سشن / کیل‌زون ──"
input bool     UseSessionFilter  = false;
input int      SessionUTCShift   = 0;      // آفست سرور بروکر نسبت به UTC (ساعت)
input int      KZ1Start = 3,  KZ1End = 4;   // ICT London   (UTC)
input int      KZ2Start = 10, KZ2End = 11;  // ICT NY AM    (UTC)
input int      KZ3Start = 14, KZ3End = 15;  // ICT NY PM    (UTC)

input group "── ۵) مدیریت ریسک ──"
input double   RiskPercent       = 0.25;   // ریسک هر معامله (٪ سمت خالص)
input double   MaxDailyLossPct   = 2.0;    // سقف ضرر روزانه ⇒ توقف
input double   MaxDrawdownPct    = 15.0;   // افت سرمایه ⇒ توقف کامل
input int      MaxTradesPerDay   = 6;
input double   MaxSpreadPoints   = 0;      // 0 = خودکار از ATR
input double   AutoSpreadATR     = 0.35;   // گیت اسپرد خودکار
input int      LadderStep2       = 2;      // باخت دوم ⇒ حجم ۰.۵
input int      LadderStep3       = 3;      // باخت سوم ⇒ حجم ۰.۲۵
input int      LadderStopAt      = 5;      // پنجمین باخت ⇒ توقف تا فردا
input double   MaxRiskMoneyAbs   = 0;      // 0 = بدون سقف مطلق

input group "── ۶) اجرا و گزارش ──"
input long     Magic             = 20260928;
input int      SlippagePoints    = 10;
input bool     LiveTrading       = false;  // ⚠ پیش‌فرض خاموش است
input bool     Verbose           = true;
input bool     ShowDashboard     = true;
input string   LogFilePrefix     = "esprakt";

//+------------------------------------------------------------------
//|  متغیرهای وضعیت                                                   |
//+------------------------------------------------------------------+
CTrade         m_trade;
CPositionInfo  m_pos;
CRiskEngine   *m_risk;

int      g_atrHandle = INVALID_HANDLE;
int      g_emaFastHandle = INVALID_HANDLE;
int      g_emaSlowHandle = INVALID_HANDLE;
datetime g_lastBarTime = 0;
int      g_barsSinceFill = 0;
ulong    g_lastDealTicket = 0;
bool     g_tradeDisabled = false;

//--- حالت «مسلح‌شدن» پس از دیدن جاروب
#define ST_IDLE    0
#define ST_ARMED   1
#define ST_PENDING 2
int      g_state = ST_IDLE;
int      g_dir = 0;
double   g_sweepExtreme = 0, g_sweepLevel = 0, g_sweepATR = 0;
int      g_sweepBar = 0, g_dispBar = -1, g_armedBar = 0;
datetime g_pendingExpiry = 0;

//+------------------------------------------------------------------+
//|  ابزار کمکی                                                       |
//+------------------------------------------------------------------+
int  GetPipDigits() { return(_Digits); }
bool IsNewBar()
  {
   datetime t = iTime(_Symbol, PERIOD_M1, 0);
   if(t != g_lastBarTime) { g_lastBarTime = t; return true; }
   return false;
  }

double Atr(int shift)
  {
   double b[1];
   if(CopyBuffer(g_atrHandle, 0, shift, 1, b) <= 0) return 0;
   return b[0];
  }

double EmaVal(int handle, int shift)
  {
   double b[1];
   if(CopyBuffer(handle, 0, shift, 1, b) <= 0) return 0;
   return b[0];
  }

void Log(const string msg)
  {
   if(!Verbose) return;
   string line = TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + "  " + msg;
   Print(line);
   int h = FileOpen(LogFilePrefix + "_" + _Symbol + ".csv", FILE_READ|FILE_WRITE|FILE_TSV|FILE_ANCHOR|FILE_COMMON);
   if(h != INVALID_HANDLE)
     {
      FileWriteString(h, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + "\t" + msg + "\n");
      FileClose(h);
     }
  }

//--- ساعت UTC از آفست سرور بروکر
int HourUTC()
  {
   MqlDateTime dt;
   TimeToStruct(TimeCurrent() + (datetime)(SessionUTCShift * 3600), dt);
   return dt.hour;
  }

bool InKillzone()
  {
   if(!UseSessionFilter) return true;
   int h = HourUTC();
   return (h >= KZ1Start && h < KZ1End) || (h >= KZ2Start && h < KZ2End) || (h >= KZ3Start && h < KZ3End);
  }

//+------------------------------------------------------------------+
//|  ساخت زمینهٔ معاملاتی                                              |
//+------------------------------------------------------------------+
MqlTradeContext g_ctx;      // مقادیر ثابت نماد — یک‌بار در OnInit ساخته می‌شود

MqlTradeContext MakeContext()
  {
   MqlTradeContext c;
   c.symbol        = _Symbol;
   c.digits       = _Digits;
   c.point         = _Point;
   c.volumeMin     = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   c.volumeMax     = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   c.volumeStep    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   c.stopLevelPoints = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   c.magic         = (int)Magic;
   c.slippagePoints= SlippagePoints;
   c.tickValue     = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   c.tickSize      = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   c.equity        = AccountInfoDouble(ACCOUNT_EQUITY);
   if(c.tickValue <= 0 || c.tickSize <= 0)
     {  // بعضی بروکرها SYMBOL_TRADE_TICK_VALUE را خالی می‌دهند ⇒ محاسبه از روی حجم ۱
      MqlTick tk;
      double v1 = 0;
      if(SymbolInfoTick(_Symbol, tk) &&
         OrderCalcProfit(ORDER_TYPE_BUY, _Symbol, 1.0, tk.bid, tk.bid + 10.0 * _Point, v1) && v1 > 0)
        c.tickValue = v1 / 10.0;
     }
   return c;
  }

//--- سوگیری جهت از میانگین‌های متحرک: +1 صعودی، -1 نزولی، 0 خنثی
int HtfBias(const int shift)
  {
   if(!UseHtfBias) return 0;
   double f = EmaVal(g_emaFastHandle, shift), s = EmaVal(g_emaSlowHandle, shift);
   double c = iClose(_Symbol, PERIOD_M1, shift);
   if(f <= 0 || s <= 0) return 0;
   if(f > s && c > f) return +1;
   if(f < s && c < f) return -1;
   return 0;
  }

//+------------------------------------------------------------------+
//|  آشکارساز سوئینگ و استخر نقدشوندگی                                |
//+------------------------------------------------------------------+
//--- سوئینگ فراکتالی: کندل b سقف است اگر از تمام k کندلِ قبل و بعد
//    خودش بلندتر باشد. چنین سوئینگی در کندل b+k *تأیید* می‌شود، نه زودتر.
bool IsSwingHigh(const int b)
  {
   if(b - SwingK < 0 || b + SwingK >= Bars) return false;
   double h = iHigh(_Symbol, PERIOD_M1, b);
   for(int i = 1; i <= SwingK; i++)
     {
      if(iHigh(_Symbol, PERIOD_M1, b + i) > h) return false;
      if(iHigh(_Symbol, PERIOD_M1, b - i) > h) return false;
     }
   return true;
  }

bool IsSwingLow(const int b)
  {
   if(b - SwingK < 0 || b + SwingK >= Bars) return false;
   double l = iLow(_Symbol, PERIOD_M1, b);
   for(int i = 1; i <= SwingK; i++)
     {
      if(iLow(_Symbol, PERIOD_M1, b + i) < l) return false;
      if(iLow(_Symbol, PERIOD_M1, b - i) < l) return false;
     }
   return true;
  }

double MaxHigh(int from, int to)
  { return iHigh(_Symbol, PERIOD_M1, iMax(from, to)); }
double MinLow (int from, int to)
  { return iLow (_Symbol, PERIOD_M1, iMin(from, to)); }

//--- استخر نقدشوندگی: میانگینِ سوئینگ‌های هم‌سطحِ *تأییدشده* تا این لحظه.
//    lastBar = آخرین کندلِ بسته (شیفت 1). آخرین کندلی که می‌تواند
//    تأییدشده باشد، lastBar - SwingK است ⇒ بدون آینده‌نگری.
bool LiquidityLevel(const int dir, const double tol, double &level, const int lastBar)
  {
   int newest = lastBar - SwingK;
   if(newest < SwingK + 1) return false;
   double pivots[];
   ArrayResize(pivots, 0);
   int look = PoolWindow + SwingK;
   for(int b = newest; b >= MathMax(SwingK + 1, newest - look); b--)
     {
      double p = 0;
      if(dir > 0) { if(!IsSwingHigh(b)) continue; p = iHigh(_Symbol, PERIOD_M1, b); }
      else        { if(!IsSwingLow(b))  continue; p = iLow (_Symbol, PERIOD_M1, b); }
      int n = ArraySize(pivots);
      ArrayResize(pivots, n + 1);
      pivots[n] = p;
     }
   if(ArraySize(pivots) < 2) return false;
   double last = pivots[0], sum = last;
   int cnt = 1;
   for(int i = 1; i < ArraySize(pivots); i++)
      if(MathAbs(pivots[i] - last) <= tol) { sum += pivots[i]; cnt++; }
   if(cnt < 2) return false;
   level = sum / cnt;
   return true;
  }

//+------------------------------------------------------------------+
//|  مدیریت معاملهٔ باز                                               |
//+------------------------------------------------------------------+
void ManageOpenPosition(const bool isNewBar)
  {
   if(!m_pos.SelectByMagic(Magic)) { if(g_state == ST_PENDING) g_state = ST_IDLE; return; }
   if(g_state == ST_PENDING)          // سفارش پر شد ⇒ دیگر دنبال سیگنال تازه نیستیم
     { g_state = ST_IDLE; g_barsSinceFill = 0; }
   ulong ticket = m_pos.Ticket();
   if(isNewBar) g_barsSinceFill++;
   if(g_barsSinceFill >= MaxHoldBars)
     {
      Log(StringFormat("⏱ خروج زمانی پس از %d کندل (RR نرسید)", g_barsSinceFill));
      m_trade.PositionClose(ticket, SlippagePoints);
      g_state = ST_IDLE; g_barsSinceFill = 0;
     }
  }

bool CancelOwnPending(ulong &ticket)
  {
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      ulong t = OrderGetTicket(i);
      if(t == 0) continue;
      if(OrderGetInteger(ORDER_MAGIC) != Magic) continue;
      if((ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE) == ORDER_TYPE_BUY_LIMIT ||
         (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE) == ORDER_TYPE_SELL_LIMIT)
        { ticket = t; return m_trade.OrderDelete(t); }
     }
   ticket = 0;
   return false;
  }

bool HasOwnPosition()  { return m_pos.SelectByMagic(Magic); }
bool HasOwnPending()
  {
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      ulong t = OrderGetTicket(i);
      if(t == 0 || OrderGetInteger(ORDER_MAGIC) != Magic) continue;
      ENUM_ORDER_TYPE ot = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
      if(ot == ORDER_TYPE_BUY_LIMIT || ot == ORDER_TYPE_SELL_LIMIT) return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
//|  نمایش داشبورد                                                    |
//+------------------------------------------------------------------+
void Dashboard(const string state)
  {
   if(!ShowDashboard) { Comment(""); return; }
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   string s = "── ESPRAKT ──Liquidity Sweep M1──\n";
   s += StringFormat("نماد: %s   بروکر: %s\n", _Symbol, AccountInfoString(ACCOUNT_COMPANY));
   s += StringFormat("تعادل: %.2f | روز: %.2f%% | معاملات امروز: %d | باخت پشت‌سرهم: %d\n",
                     eq, m_risk.DailyLossPct(), m_risk.TodayTrades(), m_risk.ConsecLosses());
   s += StringFormat("حالت: %s | ATR: %.*f | اسپرد: %d واحد\n",
                     state, GetPipDigits(), Atr(1), (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD));
   if(m_risk.IsHalted()) s += "⛔ " + m_risk.Reason();
   if(g_tradeDisabled)  s += "\n⛔ معامله‌ی واقعی غیرفعال است (LiveTrading=false)";
   Comment(s);
  }

//+------------------------------------------------------------------+
//|  OnInit                                                            |
//+------------------------------------------------------------------+
int OnInit()
  {
   g_atrHandle = iATR(_Symbol, PERIOD_M1, AtrPeriod);
   g_emaFastHandle = iMA(_Symbol, PERIOD_M1, BiasFast, 0, MODE_EMA, PRICE_CLOSE);
   g_emaSlowHandle = iMA(_Symbol, PERIOD_M1, BiasSlow, 0, MODE_EMA, PRICE_CLOSE);
   if(g_atrHandle == INVALID_HANDLE) { Print("ATR handle failed"); return INIT_FAILED; }

   g_ctx = MakeContext();
   RiskInputs ri;
   ri.RiskPercent      = RiskPercent;
   ri.MaxDailyLossPct  = MaxDailyLossPct;
   ri.MaxDrawdownPct   = MaxDrawdownPct;
   ri.MaxTradesPerDay  = MaxTradesPerDay;
   ri.MaxOpenPositions = 1;
   double a0 = Atr(1);
   ri.MaxSpreadPoints  = (MaxSpreadPoints > 0) ? MaxSpreadPoints
                        : (a0 > 0 ? (int)MathCeil(AutoSpreadATR * a0 / _Point) : 0);
   ri.MaxRiskMoneyAbs  = MaxRiskMoneyAbs;
   ri.LadderStep2      = LadderStep2;
   ri.LadderStep3      = LadderStep3;
   ri.LadderStopAt     = LadderStopAt;
   m_risk = new CRiskEngine(ri, g_ctx, LogFilePrefix);
   m_risk.Init();

   m_trade.SetExpertMagicNumber(Magic);
   m_trade.SetDeviationInPoints(SlippagePoints);
   m_trade.SetTypeFillingBySymbol(_Symbol);
   m_trade.LogLevel(LOG_LEVEL_ERRORS);

   g_lastBarTime = iTime(_Symbol, PERIOD_M1, 0);
   Log("── ربات اسپرکت راه‌اندازی شد ──");
   if(LiveTrading) Log("⚠ حالت معاملهٔ واقعی فعال است.");
   else            Log("حالت مشاهده (LiveTrading=false) — هیچ سفارشی ارسال نمی‌شود.");
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason) { if(m_risk != NULL) delete m_risk; Comment(""); }

//+------------------------------------------------------------------+
//|  OnTick — حلقهٔ اصلی                                               |
//+------------------------------------------------------------------+
void OnTick()
  {
   bool newBar = IsNewBar();
   if(newBar)
     {
      static datetime lastDay = 0;
      datetime today = StringGetTime("YYYYMMDD", TimeCurrent());
      if(today != lastDay) { lastDay = today; if(m_risk != NULL) m_risk.OnNewDay(); }
     }
   if(m_risk != NULL)
     {
      m_risk.UpdateEquity(AccountInfoDouble(ACCOUNT_EQUITY));
      m_risk.Update();
     }
   ManageOpenPosition(newBar);
   if(!newBar) return;          // داشبورد فقط روی کندل تازه بازسازی می‌شود

   //── ۰) در حالت فقط-مشاهده، فقط تحلیل می‌کنیم
   int bar = 1;                       // کندل بسته‌شده
   double a = Atr(bar);
   if(a <= 0) return;

   //── ۱) انقضای سفارش معلق
   if(g_state == ST_PENDING && HasOwnPending())
     {
      if(TimeCurrent() > g_pendingExpiry)
        { ulong t; if(CancelOwnPending(t)) Log("⌛ سفارش لیمیت منقضی شد."); g_state = ST_IDLE; }
     }

   //── ۲) تشخیص جاروب جدید روی کندل بسته
   if(g_state == ST_IDLE && !HasOwnPosition() && !HasOwnPending())
     {
      double tol = PoolTolATR * a;
      double lvl;
      // جاروب نزولی: سقف استخر شکست و زیر آن بسته شد
      if(LiquidityLevel(+1, tol, lvl, bar) &&
         iHigh(_Symbol,PERIOD_M1,bar) > lvl && iClose(_Symbol,PERIOD_M1,bar) < lvl)
        { ArmSetup(-1, lvl, iHigh(_Symbol,PERIOD_M1,bar), bar, a); }
      else if(LiquidityLevel(-1, tol, lvl, bar) &&
              iLow(_Symbol,PERIOD_M1,bar) < lvl && iClose(_Symbol,PERIOD_M1,bar) > lvl)
        { ArmSetup(+1, lvl, iLow(_Symbol,PERIOD_M1,bar), bar, a); }
     }

   //── ۳) پیشروی زنجیره: جابه‌جایی ← شکست ساختار ← ورود
   if(g_state == ST_ARMED)
     {
      if(bar - g_sweepBar > SetupMaxAge) { g_state = ST_IDLE; }
      else TryExecute();
     }
   Dashboard(g_state == ST_IDLE ? "انتظار" : (g_state == ST_ARMED ? "مسلح" : "سفارش معلق"));
  }

//+------------------------------------------------------------------+
//|  مسلح کردن ستاپ                                                   |
//+------------------------------------------------------------------+
void ArmSetup(const int dir, const double level, const double extreme, const int bar, const double a)
  {
   if(dir > 0 && !AllowLong)  return;
   if(dir < 0 && !AllowShort) return;
   if(!InKillzone()) return;
   int bias = HtfBias(bar);
   if(bias != 0 && bias != dir) return;   // خلاف جهتِ زمینهٔ تایم بالا ⇒ حذف
   g_dir = dir; g_sweepLevel = level; g_sweepExtreme = extreme;
   g_sweepBar = bar; g_sweepATR = a; g_dispBar = -1;
   g_armedBar    = iTime(_Symbol, PERIOD_M1, bar);
   g_pendingExpiry = TimeCurrent() + ExpiryMinutes * 60; // مهلت سفارش لیمیت
   g_state = ST_ARMED;
   Log(StringFormat("🔎 جاروب %s در %s — سطح %.5f، extremum %.5f",
         dir > 0 ? "خ��یدی‌سنگ" : "فروشی‌سنگ",
         TimeToString(g_armedBar, TIME_DATE|TIME_MINUTES), level, extreme));
  }

//+------------------------------------------------------------------+
//|  بررسی جابه‌جایی و شکست ساختار، سپس ارسال سفارش                  |
//+------------------------------------------------------------------+
void TryExecute()
  {
   int bar = 1;
   double a = Atr(bar);
   if(a <= 0) return;

   // گیت‌های پیش از ورود
   if(!m_risk.CanOpen()) { g_state = ST_IDLE; return; }
   double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;
   if(!m_risk.SpreadOk(spread)) { g_state = ST_IDLE; return; }

   // ③ جابه‌جایی
   if(g_dispBar < 0)
     {
      for(int b = bar; b >= MathMax(g_sweepBar, bar - DispMaxBars); b--)
        {
         double body = iClose(_Symbol,PERIOD_M1,b) - iOpen(_Symbol,PERIOD_M1,b);
         if(body * g_dir >= DispATR * Atr(b)) { g_dispBar = iTime(_Symbol,PERIOD_M1,b); break; }
        }
      if(g_dispBar < 0) return;             // هنوز جابه‌جایی نیست ⇒ صبر
     }

   // ④ شکست ساختار
   if(UseMSS)
     {
      int ref = g_sweepBar;
      double refLevel = (g_dir > 0) ? MaxHigh(ref - MssLookback, ref - 1) : MinLow(ref - MssLookback, ref - 1);
      bool broken = false;
      for(int b = bar; b >= MathMax(g_sweepBar, bar - MssMaxBars); b--)
        {
         double c = iClose(_Symbol,PERIOD_M1,b);
         if((g_dir > 0 && c > refLevel) || (g_dir < 0 && c < refLevel)) { broken = true; break; }
        }
      if(!broken) return;                   // هنوز تأیید نشده ⇒ صبر
     }

   // ⑤ سطح ورود
   int db = iBarShift(_Symbol, PERIOD_M1, g_dispBar);
   if(db < 0) { g_state = ST_IDLE; return; }
   double entry;
   bool gotFvg = false;
   if(UseFvgEntry && db >= 2)
     {
      double top, bot;
      if(g_dir > 0 && iLow(_Symbol,PERIOD_M1,db) > iHigh(_Symbol,PERIOD_M1,db-2))
        { top = iLow(_Symbol,PERIOD_M1,db); bot = iHigh(_Symbol,PERIOD_M1,db-2); }
      else if(g_dir < 0 && iHigh(_Symbol,PERIOD_M1,db) < iLow(_Symbol,PERIOD_M1,db-2))
        { top = iLow(_Symbol,PERIOD_M1,db-2); bot = iHigh(_Symbol,PERIOD_M1,db); }
      else { g_state = ST_IDLE; return; }
      if((top - bot) < FvgMinATR * a) { g_state = ST_IDLE; return; }
      entry = bot + (top - bot) * FvgEntryPos;
      gotFvg = true;
     }
   if(!gotFvg)
      entry = iLow(_Symbol,PERIOD_M1,db) + (iHigh(_Symbol,PERIOD_M1,db) - iLow(_Symbol,PERIOD_M1,db)) * 0.5;

   // ⑥ استاپ ساختاری
   double stop = g_sweepExtreme - g_dir * StopBufferATR * g_sweepATR;
   double risk = MathAbs(entry - stop);
   if(risk <= 0) { g_state = ST_IDLE; return; }
   if(risk < MinStopATR * a || risk > MaxStopATR * a) { g_state = ST_IDLE; return; }
   if(risk < MinStopSpread * spread) { g_state = ST_IDLE; return; }

   // لیمیت باید واقعاً «لیمیت» باشد: خرید زیرِ بازار، فروش بالای بازار
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(g_dir > 0 && entry >= ask) { g_state = ST_IDLE; return; }
   if(g_dir < 0 && entry <= bid) { g_state = ST_IDLE; return; }

   // حجم بر پایهٔ فاصلهٔ واقعیِ استاپ تا قیمتِ بازار (نه تا حد)
   double realRisk = (g_dir > 0) ? (ask - stop) : (stop - bid);
   if(realRisk <= 0) { g_state = ST_IDLE; return; }
   double lots = m_risk.LotsFor(realRisk, g_ctx.tickValue, g_ctx.tickSize);
   if(lots <= 0) { g_state = ST_IDLE; return; }

   double target = (g_dir > 0) ? ask + RR * realRisk : bid - RR * realRisk;
   ENUM_ORDER_TYPE ot = (g_dir > 0) ? ORDER_TYPE_BUY_LIMIT : ORDER_TYPE_SELL_LIMIT;
   double price = NormalizeDouble(entry, _Digits);
   double sl = NormalizeDouble(stop, _Digits);
   double tp = NormalizeDouble(target, _Digits);

   string note = StringFormat("%s | ورود %.5f | استاپ %.5f | هدف %.5f | R=%.2f | حجم %.2f",
         gotFvg ? "FVG" : "وسط‌کندل", price, sl, tp, MathAbs(price-stop), lots);
   if(!LiveTrading)
     {
      Log("📝 [شبیه‌سازی] " + note);
      g_state = ST_IDLE;
      return;
     }
   if(m_trade.PlaceLimit(ot, price, lots, sl, tp, ORDER_TIME_SPECIFIED, g_pendingExpiry))
      { Log("✅ سفارش ثبت شد: " + note); g_state = ST_PENDING; }
   else
      { Log("❌ خطای ثبت سفارش: " + StringFormat("%d", GetLastError())); g_state = ST_IDLE; }
  }

//+------------------------------------------------------------------+
//|  OnTradeTransaction — پایش نتیجه برای نردبان برق‌گیر               |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
  {
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
   if(!HistoryDealSelect(trans.deal)) return;
   if(HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != Magic) return;
   if(HistoryDealGetInteger(trans.deal, DEAL_ENTRY) != DEAL_ENTRY_OUT) return;
   double profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT)
                 + HistoryDealGetDouble(trans.deal, DEAL_SWAP)
                 + HistoryDealGetDouble(trans.deal, DEAL_COMMISSION);
   m_risk.RegisterResult(profit > 0);
   g_barsSinceFill = 0;
   g_state = ST_IDLE;
   Log(StringFormat("🔚 بسته شد: %+.2f (%s) — باخت‌های پشت‌سرهم: %d",
         profit, profit > 0 ? "برد" : "باخت", m_risk.ConsecLosses()));
  }
//+------------------------------------------------------------------+
