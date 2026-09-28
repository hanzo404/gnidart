//+------------------------------------------------------------------+
//|  Esprakt_RiskEngine.mqh                                           |
//|  موتور ریسک: سایزینگ درصدی، سقف ضرر روزانه، نردبان برق‌گیر، گیت اسپرد|
//+------------------------------------------------------------------+
/*
   چرا این فایل جدا است؟

   آزمون مستقلِ ۱۰ ساله‌ای که در پژوهش این ربات دیده شد، نشان داد یک ستاپ
   با ۶۱٪ نرخ برد و ۱:۲.۴ ریسک‌به‌بازده، باز هم ۱۰۰٬۰۰۰ دلار را به ۵۴٬۰۰۰
   تبدیل کرد. یعنی «نرخ بردِ بالا» تضمینِ سودآوری نیست؛ آنچه تضمین می‌کند
   کنترل نوسان است. کل ارزش این ماژول در همین است، نه در تشخیص سیگنال.

   اصولی که اینجا رعایت می‌شوند و از دلِ ریپوی خودمان (STATE_AND_PATHS.md)
   و کتابخانهٔ ۱۲۸ کتابی استخراج شده‌اند:
     • ریسک ثابت و درصدی — نه مارتینگل، نه انباشت، هرگز
     • سقف ضرر روزانه: پس از آن دیگر معامله‌ای باز نمی‌شود
     • نردبان برق‌گیر: باخت‌های پشت‌سرهم ⇒ حجم کمتر (نه تغییر استراتژی)
     • گیت اسپرد: اگر اسپرد از حد گذشت، ستاپ اجرا نمی‌شود
*/
//+------------------------------------------------------------------+
#property copyright "gnidart research"
#property strict

//+------------------------------------------------------------------+
//|  زمینهٔ معاملاتی: هر چیزی که موتور ریسک برای تصمیم لازم دارد.       |
//+------------------------------------------------------------------+
struct MqlTradeContext
  {
   string   symbol;
   int      digits;
   double   point;
   double   tickValue;      // ارزش ۱ تیک برای ۱ لات
   double   tickSize;       // اندازهٔ ۱ تیک
   double   volumeMin, volumeMax, volumeStep;
   double   stopLevelPoints;// حداقل فاصلهٔ استاپ/لیمیت که بروکر اجازه می‌دهد
   double   equity;         // با هر تیک به‌روز می‌شود (منبع حقیقت موتور ریسک)
   int      magic;
   int      slippagePoints;
  };
//+------------------------------------------------------------------+
struct RiskInputs
  {
   double RiskPercent;        // ریسک هر معامله (درصدی از سمت خالص)
   double MaxDailyLossPct;    // سقف ضرر روزانه قبل از توقف
   double MaxDrawdownPct;     // افت سرمایه از سقف ⇒ توقف کامل
   int    MaxTradesPerDay;
   int    MaxOpenPositions;
   double MaxSpreadPoints;    // گیت اسپرد (واحد پوینت نماد)
   double MaxRiskMoneyAbs;    // سقف مطلق ریسک هر معامله (پول حساب)
   int    LadderStep2;        // پس از این باختِ پشت‌سرهم
   int    LadderStep3;
   int    LadderStopAt;
  };
//+------------------------------------------------------------------+

class CRiskEngine
  {
private:
   RiskInputs      m_in;
   MqlTradeContext m_ctx;
   string          m_prefix;
   datetime        m_dayStart;
   double          m_dayStartEquity;
   double          m_peakEquity;
   int             m_todayTrades;
   int             m_consecLosses;
   bool            m_haltedDay;
   bool            m_haltedAll;
   string          m_reason;

public:
                     CRiskEngine(const RiskInputs &inp, const MqlTradeContext &ctx, const string prefix) { m_in=inp; m_ctx=ctx; m_prefix=prefix; }

   void             Init()
     {
      m_dayStart      = 0;
      m_dayStartEquity= m_ctx.equity;
      m_peakEquity    = m_ctx.equity;
      m_todayTrades   = 0;
      m_consecLosses  = 0;
      m_haltedDay     = false;
      m_haltedAll     = false;
      m_reason        = "";
     }

   //--- روز جدید ⇒ صفر کردن شمارنده‌های روزانه
   void             OnNewDay()
     {
      m_dayStart       = TimeCurrent();
      m_dayStartEquity = m_ctx.equity;
      m_todayTrades    = 0;
      m_haltedDay      = false;
     }

   //--- تعادلِ زنده باید هر تیک به‌روز شود (وگرنه موتور روی عدد کهن تصمیم می‌گیرد)
   void             UpdateEquity(const double eq) { m_ctx.equity = eq; }

   //--- به‌روزرسانی با هر تیک (سبک)
   void             Update()
     {
      if(m_peakEquity < m_ctx.equity) m_peakEquity = m_ctx.equity;
      if(!m_haltedAll && m_peakEquity > 0 &&
         (m_peakEquity - m_ctx.equity) / m_peakEquity * 100.0 >= m_in.MaxDrawdownPct)
        {
         m_haltedAll = true;
         m_reason    = StringFormat("توقف کامل: افت سرمایه %.1f%% ≥ %.1f%%",
                                    (m_peakEquity-m_ctx.equity)/m_peakEquity*100.0, m_in.MaxDrawdownPct);
        }
     }

   //--- پس از بسته شدن هر معامله
   void             RegisterResult(const bool win)
     {
      m_todayTrades++;
      if(win) m_consecLosses = 0;
      else    m_consecLosses++;
     }

   double           DailyLossPct() const
     {
      if(m_dayStartEquity <= 0) return 0;
      double p = (m_dayStartEquity - m_ctx.equity) / m_dayStartEquity * 100.0;
      return p;
     }

   int              ConsecLosses() const { return m_consecLosses; }
   bool             IsHalted()    const { return m_haltedAll || m_haltedDay; }
   string           Reason()      const { return m_reason; }
   double           PeakEquity()  const { return m_peakEquity; }
   int              TodayTrades() const { return m_todayTrades; }

   //--- آیا اجازهٔ باز کردن معاملهٔ تازه هست؟
   bool             CanOpen() const
     {
      if(m_haltedAll || m_haltedDay)          return false;
      if(m_todayTrades >= m_in.MaxTradesPerDay) return false;
      if(DailyLossPct() >= m_in.MaxDailyLossPct) return false;
      if(SizeMultiplier() <= 0)               return false;  // نردبان برق‌گیر اجازه نداده
      return true;
     }

   //--- ضریب حجم از نردبان برق‌گیر
   double           SizeMultiplier() const
     {
      if(m_consecLosses >= m_in.LadderStopAt) return 0.0;
      if(m_consecLosses >= m_in.LadderStep3)  return 0.25;
      if(m_consecLosses >= m_in.LadderStep2)  return 0.50;
      return 1.0;
     }

   //--- گیت اسپرد: بازارِ گران، بازی ممنوع
   bool             SpreadOk(const double askPoints) const
     { return (m_in.MaxSpreadPoints <= 0) || (askPoints <= m_in.MaxSpreadPoints); }

   //--- محاسبهٔ حجم بر پایهٔ فاصلهٔ استاپ
   double           LotsFor(const double stopDistancePrice, const double tickValue, const double tickSize) const
     {
      double mult = SizeMultiplier();
      if(mult <= 0) return 0.0;
      double riskMoney = m_ctx.equity * m_in.RiskPercent / 100.0 * mult;
      if(riskMoney > m_in.MaxRiskMoneyAbs && m_in.MaxRiskMoneyAbs > 0)
         riskMoney = m_in.MaxRiskMoneyAbs;
      if(stopDistancePrice <= 0 || tickValue <= 0 || tickSize <= 0) return 0.0;
      double lossPerLot = (stopDistancePrice / tickSize) * tickValue;
      if(lossPerLot <= 0) return 0.0;
      double lots = riskMoney / lossPerLot;
      return NormalizeVolume(lots);
     }

   double           NormalizeVolume(const double lots) const
     {
      double mn = m_ctx.symbolVolMin, mx = m_ctx.symbolVolMax, step = m_ctx.symbolVolStep;
      if(step <= 0) step = 0.01;
      double v = MathFloor(lots / step + 0.5) * step;
      if(v < mn) v = mn;
      if(v > mx) v = mx;
      // اگر بعد از گِرد کردن به کف چسبید و ریسک از حد گذشت ⇒ حذف معامله
      return NormalizeDouble(v, 2);
     }
  };
//+------------------------------------------------------------------+
